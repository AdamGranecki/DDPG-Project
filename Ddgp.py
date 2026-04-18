"""
DDPG.py — Deep Deterministic Policy Gradient
Implementacja dla środowiska PyBullet z robotem UR5.

Architektura:
  Actor:  obs → akcje (delta kątów stawów)
  Critic: (obs, akcje) → Q-wartość
  Replay Buffer: doświadczenia (s, a, r, s', done)
  Szum OU: eksploracja ciągłej przestrzeni akcji
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import collections
import random
import os
import matplotlib.pyplot as plt


# ─────────────────────────────────────────────
#  KONFIGURACJA
# ─────────────────────────────────────────────

class DDPGConfig:
    # Sieć
    ACTOR_HIDDEN   = [256, 256]
    CRITIC_HIDDEN  = [256, 256]

    # Trening
    GAMMA          = 0.99       # współczynnik dyskontowania
    TAU            = 0.005      # miękka aktualizacja sieci docelowych
    ACTOR_LR       = 1e-4
    CRITIC_LR      = 1e-3
    BATCH_SIZE     = 64
    BUFFER_SIZE    = 100_000
    WARMUP_STEPS   = 1000       # losowe akcje przed startem uczenia

    # Szum OU
    OU_MU          = 0.0
    OU_THETA       = 0.15
    OU_SIGMA       = 0.2
    OU_SIGMA_DECAY = 0.9995     # sigma maleje co epizod
    OU_SIGMA_MIN   = 0.01

    # Granice akcji (delta kąta na krok dla każdego stawu [rad])
    ACTION_SCALE   = 0.05

    # Zapis
    SAVE_DIR       = "checkpoints"
    PLOT_EVERY     = 10         # rysuj wykres co N epizodów


# ─────────────────────────────────────────────
#  SZUM ORNSTEINA-UHLENBECKA
# ─────────────────────────────────────────────

class OUNoise:
    """Szum OU – korelowany czasowo, dobry do eksploracji w ciągłej przestrzeni."""

    def __init__(self, size, mu=0.0, theta=0.15, sigma=0.2):
        self.mu    = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.size  = size
        self.reset()

    def reset(self):
        self.state = np.copy(self.mu)

    def sample(self):
        dx = self.theta * (self.mu - self.state) + \
             self.sigma * np.random.randn(self.size)
        self.state += dx
        return self.state


# ─────────────────────────────────────────────
#  REPLAY BUFFER
# ─────────────────────────────────────────────

class ReplayBuffer:
    def __init__(self, capacity=100_000):
        self.buffer = collections.deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.FloatTensor(np.array(states)),
            torch.FloatTensor(np.array(actions)),
            torch.FloatTensor(np.array(rewards)).unsqueeze(1),
            torch.FloatTensor(np.array(next_states)),
            torch.FloatTensor(np.array(dones)).unsqueeze(1),
        )

    def __len__(self):
        return len(self.buffer)


# ─────────────────────────────────────────────
#  SIECI NEURONOWE
# ─────────────────────────────────────────────

def _mlp(in_dim, hidden_dims, out_dim, activation=nn.ReLU, out_activation=None):
    layers = []
    prev = in_dim
    for h in hidden_dims:
        layers += [nn.Linear(prev, h), activation()]
        prev = h
    layers.append(nn.Linear(prev, out_dim))
    if out_activation is not None:
        layers.append(out_activation())
    return nn.Sequential(*layers)


class Actor(nn.Module):
    """
    Mapuje obserwację → akcję (delta kątów stawów).
    Wyjście ∈ [-1, 1] (Tanh), skalowane przez ACTION_SCALE.
    """
    def __init__(self, obs_dim, act_dim, hidden=[256, 256]):
        super().__init__()
        self.net = _mlp(obs_dim, hidden, act_dim, out_activation=nn.Tanh)

    def forward(self, obs):
        return self.net(obs)


class Critic(nn.Module):
    """
    Mapuje (obserwację, akcję) → wartość Q.
    """
    def __init__(self, obs_dim, act_dim, hidden=[256, 256]):
        super().__init__()
        self.net = _mlp(obs_dim + act_dim, hidden, 1)

    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=-1)
        return self.net(x)


# ─────────────────────────────────────────────
#  AGENT DDPG
# ─────────────────────────────────────────────

class DDPGAgent:
    """
    Główna klasa DDPG.

    Wymiary:
      obs_dim  – rozmiar wektora obserwacji
      act_dim  – liczba stawów (6 dla UR5)
    """

    def __init__(self, obs_dim, act_dim, cfg: DDPGConfig = None):
        self.cfg     = cfg or DDPGConfig()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[DDPG] Urządzenie: {self.device}")

        # ── Sieci (online + docelowe) ──
        self.actor        = Actor(obs_dim, act_dim, self.cfg.ACTOR_HIDDEN).to(self.device)
        self.actor_target = Actor(obs_dim, act_dim, self.cfg.ACTOR_HIDDEN).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic        = Critic(obs_dim, act_dim, self.cfg.CRITIC_HIDDEN).to(self.device)
        self.critic_target = Critic(obs_dim, act_dim, self.cfg.CRITIC_HIDDEN).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # ── Optymalizatory ──
        self.actor_opt  = optim.Adam(self.actor.parameters(),  lr=self.cfg.ACTOR_LR)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=self.cfg.CRITIC_LR)

        # ── Replay buffer ──
        self.buffer = ReplayBuffer(self.cfg.BUFFER_SIZE)

        # ── Szum OU ──
        self.noise       = OUNoise(act_dim, self.cfg.OU_MU, self.cfg.OU_THETA, self.cfg.OU_SIGMA)
        self.ou_sigma    = self.cfg.OU_SIGMA

        # ── Statystyki ──
        self.total_steps   = 0
        self.episode_rewards = []
        self.critic_losses   = []
        self.actor_losses    = []

        os.makedirs(self.cfg.SAVE_DIR, exist_ok=True)

    # ── Wybór akcji ──────────────────────────────────────────────────────────

    def select_action(self, obs, explore=True):
        """
        Zwraca deltę kątów stawów ∈ [-ACTION_SCALE, ACTION_SCALE].
        explore=True podczas treningu, False podczas ewaluacji.
        """
        if explore and self.total_steps < self.cfg.WARMUP_STEPS:
            # Faza rozgrzewki – losowe akcje
            return np.random.uniform(-1, 1, self.act_dim) * self.cfg.ACTION_SCALE

        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action = self.actor(obs_t).cpu().numpy()[0]

        if explore:
            noise  = self.noise.sample()
            action = np.clip(action + noise, -1.0, 1.0)

        return action * self.cfg.ACTION_SCALE

    # ── Zapis doświadczenia ───────────────────────────────────────────────────

    def remember(self, state, action, reward, next_state, done):
        # Normalizuj akcję z powrotem do [-1, 1] przed zapisem
        self.buffer.push(state, action / self.cfg.ACTION_SCALE, reward, next_state, done)
        self.total_steps += 1

    # ── Krok uczenia ─────────────────────────────────────────────────────────

    def learn(self):
        """
        Jeden krok aktualizacji sieci.
        Zwraca (critic_loss, actor_loss) lub None jeśli za mało danych.
        """
        if len(self.buffer) < self.cfg.BATCH_SIZE:
            return None

        states, actions, rewards, next_states, dones = \
            self.buffer.sample(self.cfg.BATCH_SIZE)

        states      = states.to(self.device)
        actions     = actions.to(self.device)
        rewards     = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones       = dones.to(self.device)

        # ── Aktualizacja Critika ──────────────────────────────────────────
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            q_target     = rewards + self.cfg.GAMMA * (1 - dones) * \
                           self.critic_target(next_states, next_actions)

        q_pred      = self.critic(states, actions)
        critic_loss = nn.MSELoss()(q_pred, q_target)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # ── Aktualizacja Aktora ───────────────────────────────────────────
        pred_actions = self.actor(states)
        actor_loss   = -self.critic(states, pred_actions).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # ── Miękka aktualizacja sieci docelowych (Polyak) ────────────────
        self._soft_update(self.actor,  self.actor_target)
        self._soft_update(self.critic, self.critic_target)

        return critic_loss.item(), actor_loss.item()

    def _soft_update(self, source, target):
        tau = self.cfg.TAU
        for src_p, tgt_p in zip(source.parameters(), target.parameters()):
            tgt_p.data.copy_(tau * src_p.data + (1 - tau) * tgt_p.data)

    # ── Koniec epizodu ────────────────────────────────────────────────────────

    def end_episode(self, total_reward, episode_num):
        self.episode_rewards.append(total_reward)
        self.noise.reset()

        # Zmniejsz szum eksploracji
        self.ou_sigma = max(
            self.cfg.OU_SIGMA_MIN,
            self.ou_sigma * self.cfg.OU_SIGMA_DECAY
        )
        self.noise.sigma = self.ou_sigma

        print(f"[Ep {episode_num:4d}] Nagroda: {total_reward:8.2f} | "
              f"Bufor: {len(self.buffer):6d} | Sigma OU: {self.ou_sigma:.4f}")

        if episode_num % self.cfg.PLOT_EVERY == 0:
            self.plot_rewards()

    # ── Zapis i wczytywanie ───────────────────────────────────────────────────

    def save(self, episode):
        path = os.path.join(self.cfg.SAVE_DIR, f"ddpg_ep{episode}.pt")
        torch.save({
            "actor":        self.actor.state_dict(),
            "critic":       self.critic.state_dict(),
            "actor_target": self.actor_target.state_dict(),
            "critic_target":self.critic_target.state_dict(),
            "actor_opt":    self.actor_opt.state_dict(),
            "critic_opt":   self.critic_opt.state_dict(),
            "episode_rewards": self.episode_rewards,
            "total_steps":  self.total_steps,
            "ou_sigma":     self.ou_sigma,
        }, path)
        print(f"[DDPG] Zapisano model → {path}")

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.actor_target.load_state_dict(ckpt["actor_target"])
        self.critic_target.load_state_dict(ckpt["critic_target"])
        self.actor_opt.load_state_dict(ckpt["actor_opt"])
        self.critic_opt.load_state_dict(ckpt["critic_opt"])
        self.episode_rewards = ckpt.get("episode_rewards", [])
        self.total_steps     = ckpt.get("total_steps", 0)
        self.ou_sigma        = ckpt.get("ou_sigma", self.cfg.OU_SIGMA)
        self.noise.sigma     = self.ou_sigma
        print(f"[DDPG] Wczytano model ← {path}")

    # ── Wykres nagród ─────────────────────────────────────────────────────────

    def plot_rewards(self):
        rewards = np.array(self.episode_rewards)
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(rewards, alpha=0.4, color="steelblue", label="Nagroda")

        # Średnia krocząca (okno 20 epizodów)
        if len(rewards) >= 20:
            ma = np.convolve(rewards, np.ones(20)/20, mode="valid")
            ax.plot(range(19, len(rewards)), ma, color="tomato", lw=2, label="Średnia krocząca (20)")

        ax.set_xlabel("Epizod")
        ax.set_ylabel("Łączna nagroda")
        ax.set_title("DDPG – Krzywa uczenia (UR5 Reach)")
        ax.legend()
        fig.tight_layout()
        plt.savefig(os.path.join(self.cfg.SAVE_DIR, "learning_curve.png"), dpi=120)
        plt.close()
        print("[DDPG] Wykres zapisany → checkpoints/learning_curve.png")