"""
Main.py — Punkt wejścia: trening i ewaluacja agenta DDPG na robocie UR5.

Tryby uruchomienia:
  python Main.py              → trening od zera
  python Main.py --eval       → ewaluacja wczytanego modelu (bez uczenia)
  python Main.py --load PATH  → wznowienie treningu z checkpointu
"""

import argparse
from Enviroment import Environment
from Agent import DDPGRobotAgent


# ─────────────────────────────────────────────
#  PARAMETRY
# ─────────────────────────────────────────────

TARGET_POS    = [0.3, -0.5, 1.0]   # pozycja TCP do osiągnięcia [m]
NUM_EPISODES  = 500                 # liczba epizodów treningu
MAX_STEPS     = 300                 # maksymalna liczba kroków w epizodzie
EPS_REACH     = 0.05                # [m] tolerancja osiągnięcia celu


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval",  action="store_true",
                        help="Tryb ewaluacji – brak uczenia, brak szumu")
    parser.add_argument("--load",  type=str, default=None,
                        help="Ścieżka do checkpointu .pt")
    parser.add_argument("--episodes", type=int, default=NUM_EPISODES)
    args = parser.parse_args()

    training = not args.eval

    # ── Środowisko ──
    env = Environment(gui=True)

    # ── Agent ──
    agent = DDPGRobotAgent(
        env            = env,
        target_pos     = TARGET_POS,
        eps_reach      = EPS_REACH,
        max_steps      = MAX_STEPS,
        load_checkpoint= args.load,
        training       = training,
    )

    print("=" * 60)
    print(f"  Tryb: {'TRENING' if training else 'EWALUACJA'}")
    print(f"  Cel:  {TARGET_POS}")
    print(f"  Epizody: {args.episodes}")
    print("=" * 60)

    # ── Pętla epizodów ──
    for ep in range(1, args.episodes + 1):
        total_reward, steps = agent.run_episode()
        print(f"Epizod {ep:4d}/{args.episodes} | "
              f"Nagroda: {total_reward:8.2f} | Kroki: {steps}")

    # ── Końcowy zapis ──
    if training:
        agent.ddpg.save(episode=args.episodes)
        agent.ddpg.plot_rewards()
        print("\n✅ Trening zakończony. Model i wykres zapisane w ./checkpoints/")

    env.close()


if __name__ == "__main__":
    main()