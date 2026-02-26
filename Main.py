import pybullet_data
import time
import math
from collections import namedtuple
import numpy as np
import pybullet as p
from Agent import Agent
from Enviroment import Environment
from Agent import Agent


# Pętla symulacji
env = Environment(gui=True)

agent = Agent(
    env=env,
    target_pos=[.3,-.5,1],  # pozycja B
    eps=0.1
)

while True:
    env.step()      # symulacja + hold
    # agent.step()    # logika ruchu + zdjęcie

    if agent.done:
        print("🛑 Zatrzymanie programu")
        break

time.sleep(10)
env.close()

