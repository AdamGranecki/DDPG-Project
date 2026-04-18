"""
Agent.py — Agent RL sterujący robotem UR5 przy pomocy DDPG.

Obserwacja (obs_dim = 15):
  - pozycja TCP                    [3]
  - pozycja celu                   [3]
  - wektor błędu TCP → cel         [3]
  - kąty 6 stawów ramienia         [6]

Akcja (act_dim = 6):
  - delta kąta każdego stawu       [6]  ∈ [-ACTION_SCALE, ACTION_SCALE]

Nagroda:
  - r = -dist            (im bliżej celu, tym wyższa)
  - r += +10             (bonus za osiągnięcie celu)
  - r += -0.01           (kara za każdy krok – motywacja do szybkości)
"""

import numpy as np
import pybullet as p

from DDPG import DDPGAgent, DDPGConfig


# ─────────────────────────────────────────────
#  WYMIARY PRZESTRZENI
# ─────────────────────────────────────────────

OBS_DIM = 15   # [tcp_pos(3), target_pos(3), error(3), joint_angles(6)]
ACT_DIM = 6    # delta kątów dla 6 stawów UR5


# ─────────────────────────────────────────────
#  AGENT
# ─────────────────────────────────────────────

class DDPGRobotAgent:
    """
    Łączy środowisko PyBullet z algorytmem DDPG.

    Użycie:
        agent = DDPGRobotAgent(env, target_pos=[0.3, -0.5, 1.0])
        for ep in range(NUM_EPISODES):
            agent.run_episode(max_steps=300)
    """

    def __init__(
        self,
        env,
        target_pos,
        eps_reach      = 0.05,   # [m] dystans uznany za osiągnięcie celu
        max_steps      = 300,    # maks. kroki na epizod
        load_checkpoint= None,   # ścieżka do pliku .pt lub None
        training       = True,   # False = tryb ewaluacji (bez szumu, bez uczenia)
    ):
        self.env        = env
        self.robot      = env.ur5
        self.target_pos = np.array(target_pos, dtype=np.float32)
        self.eps_reach  = eps_reach
        self.max_steps  = max_steps
        self.training   = training

        self.cfg   = DDPGConfig()
        self.ddpg  = DDPGAgent(OBS_DIM, ACT_DIM, self.cfg)

        if load_checkpoint:
            self.ddpg.load(load_checkpoint)

        # Zapamiętaj pozycję startową stawów (reset po każdym epizodzie)
        self.start_joints = list(self.robot.start_poses)

        self.episode_num = 0

    # ── Obserwacja ────────────────────────────────────────────────────────────

    def _get_obs(self):
        # Pozycja TCP
        ls      = p.getLinkState(self.robot.robot_id, self.robot.eef_id,
                                 computeForwardKinematics=True)
        tcp_pos = np.array(ls[4], dtype=np.float32)

        # Kąty stawów
        joints  = np.array([
            p.getJointState(self.robot.robot_id, j)[0]
            for j in self.robot.arm_controllable_joints
        ], dtype=np.float32)

        error   = self.target_pos - tcp_pos

        obs = np.concatenate([tcp_pos, self.target_pos, error, joints])
        assert obs.shape == (OBS_DIM,), f"Zły rozmiar obs: {obs.shape}"
        return obs, tcp_pos

    # ── Nagroda ───────────────────────────────────────────────────────────────

    def _compute_reward(self, tcp_pos, reached):
        dist   = np.linalg.norm(self.target_pos - tcp_pos)
        reward = -dist                   # gęsty sygnał kształtujący
        reward -= 0.01                   # kara za każdy krok (czas)
        if reached:
            reward += 10.0               # bonus za osiągnięcie celu
        return float(reward)

    # ── Reset środowiska ──────────────────────────────────────────────────────

    def _reset(self):
        """Powrót ramienia do pozycji startowej."""
        self.robot.move_arm(self.start_joints, 'joint')
        # Kilka kroków symulacji żeby robot zdążył wrócić
        for _ in range(50):
            self.env.step()

    # ── Aplikacja akcji ───────────────────────────────────────────────────────

    def _apply_action(self, delta_joints):
        """
        Dodaj deltę do aktualnych kątów stawów i wyślij komendę.
        Kąty są obcinane do granic stawów.
        """
        current = np.array([
            p.getJointState(self.robot.robot_id, j)[0]
            for j in self.robot.arm_controllable_joints
        ])
        new_joints = current + delta_joints
        # Obcinanie do granic
        new_joints = np.clip(
            new_joints,
            self.robot.arm_lower_limits,
            self.robot.arm_upper_limits
        )
        self.robot.move_arm(new_joints.tolist(), 'joint')

    # ── Jeden epizod ──────────────────────────────────────────────────────────

    def run_episode(self):
        """
        Przeprowadza jeden pełny epizod.
        Zwraca łączną nagrodę i liczbę kroków.
        """
        self.episode_num += 1
        self._reset()

        obs, tcp_pos = self._get_obs()
        total_reward = 0.0

        for step in range(self.max_steps):
            # 1. Wybór akcji
            action = self.ddpg.select_action(obs, explore=self.training)

            # 2. Aplikacja akcji
            self._apply_action(action)

            # 3. Krok symulacji (kilka sub-stepów dla stabilności)
            for _ in range(4):
                self.env.step()

            # 4. Nowa obserwacja
            next_obs, tcp_pos = self._get_obs()
            dist    = np.linalg.norm(self.target_pos - tcp_pos)
            reached = dist < self.eps_reach
            done    = reached or (step == self.max_steps - 1)

            # 5. Nagroda
            reward = self._compute_reward(tcp_pos, reached)
            total_reward += reward

            # 6. Zapisz doświadczenie i ucz
            if self.training:
                self.ddpg.remember(obs, action, reward, next_obs, float(done))
                result = self.ddpg.learn()
                if result and step % 50 == 0:
                    c_loss, a_loss = result
                    print(f"  krok {step:3d} | dist={dist:.3f}m | "
                          f"C_loss={c_loss:.4f} | A_loss={a_loss:.4f}")

            obs = next_obs

            if reached:
                print(f"  ✅ Cel osiągnięty! Dystans: {dist:.4f}m (krok {step})")
                break

        # 7. Statystyki epizodu
        self.ddpg.end_episode(total_reward, self.episode_num)

        # 8. Zapis co 50 epizodów
        if self.training and self.episode_num % 50 == 0:
            self.ddpg.save(self.episode_num)

        return total_reward, step + 1