import numpy as np


# --------------------------------------------------------------------------- #
# Utilities                                                                   #
# --------------------------------------------------------------------------- #

class Namer(object):

  COUNTER = 0

  @classmethod
  def name(cls):
    next_name = str(cls.COUNTER)
    cls.COUNTER += 1
    return next_name


# --------------------------------------------------------------------------- #
# Neural network that controls beast behavior                                 #
# --------------------------------------------------------------------------- #

class Brain(object):

  LEARNING_RATE = 0.01

  def __init__(self, beast):
    # State
    self.memory = np.zeros(beast.MEMORY_NEURONS)
    self.proprioception = np.zeros(beast.ACTION_NEURONS)

    # Weights
    self.sensor_to_hidden = np.reshape(
        beast.synapses[:beast.HIDDEN_CX],
        (beast.SENSOR_NEURONS + beast.ACTION_NEURONS, beast.HIDDEN_NEURONS))
    self.hidden_to_memory = np.reshape(
        beast.synapses[beast.HIDDEN_CX:-beast.ACTION_CX],
        (beast.HIDDEN_NEURONS + beast.MEMORY_NEURONS, beast.MEMORY_NEURONS))
    self.memory_to_action = np.reshape(
        beast.synapses[-beast.ACTION_CX:],
        (beast.MEMORY_NEURONS, beast.ACTION_NEURONS))

  def learn(self, reward=1.0):
    self.memory_to_action += (
      self.memory_to_action *
      (reward * self.LEARNING_RATE * self.proprioception))

  def think(self, percept):
    # Propogate sensations into hidden layer
    hidden = np.tanh(np.dot(
        np.concatenate([percept, self.proprioception]),
        self.sensor_to_hidden))

    # Recurrent memory
    memory = np.tanh(np.dot(
        np.concatenate([hidden, self.memory]),
        self.hidden_to_memory))
    self.memory = (self.memory + memory) / 2.0

    # Choose action, and update recurrent proprioception
    action_values = np.dot(memory, self.memory_to_action)
    action = np.argmax(action_values)
    self.proprioception /= 2.0
    self.proprioception[action] += 1.0
    return action


# --------------------------------------------------------------------------- #
# Our synthetic organisms.                                                    #
# --------------------------------------------------------------------------- #

class Beast(object):

  SENSOR_NEURONS = 13
  HIDDEN_NEURONS = 10
  MEMORY_NEURONS = 10
  ACTION_NEURONS = 5  # left, right, forward, eat, reproduce

  HIDDEN_CX = HIDDEN_NEURONS * (ACTION_NEURONS + SENSOR_NEURONS)
  MEMORY_CX = MEMORY_NEURONS * (HIDDEN_NEURONS + MEMORY_NEURONS)
  ACTION_CX = ACTION_NEURONS * MEMORY_NEURONS
  TOTAL_BRAIN_CX = HIDDEN_CX + MEMORY_CX + ACTION_CX

  HEADINGS = {
      0: (1, 1),
      1: (0, 1),
      2: (-1, 1),
      3: (-1, 0),
      4: (-1, -1),
      5: (0, -1),
      6: (1, -1),
      7: (1, 0)}

  METABOLIC_COST = 0.01
  DROWNING_COST = 0.01
  MOVEMENT_COST = 0.001
  REPRODUCTION_COST = 1.0

  def __init__(self, world):
    self.world = world
    self.name = Namer.name()

    # Genome
    self.protein_hue = np.random.rand()
    self.pelt_hue = np.random.rand()
    self.synapses = np.random.normal(0, 0.01, self.TOTAL_BRAIN_CX)
    self.brain = Brain(self)

    # History
    self.age = 0
    self.generation = 0

    # Current state
    self.coords = np.array([
        np.random.randint(self.world.xsize),
        np.random.randint(self.world.ysize)])
    self.energy = 1.0
    self.heading = np.random.randint(0, 8)

  def _perceive(self):
    # Where are my sensors?
    center_x, center_y = self.coords
    left_x, left_y = self.coords + np.array(
        self.HEADINGS[(self.heading + 1) % 8])
    front_x, front_y = self.coords + np.array(self.HEADINGS[self.heading])
    right_x, right_y = self.coords + np.array(
        self.HEADINGS[(self.heading - 1) % 8])

    # Construct array of sensor inputs
    return np.array(
        self.world.percept_at(center_x, center_y) +
        self.world.percept_at(left_x, left_y) +
        self.world.percept_at(front_x, front_y) +
        self.world.percept_at(right_x, right_y) +
        (self.energy,))

  def _left(self):
    self.heading = (self.heading + 1) % 8

  def _forward(self):
    self.energy -= self.energy * self.MOVEMENT_COST
    x, y = self.coords + np.array(self.HEADINGS[self.heading])
    if not self.world.valid_coords(x, y):
      return
    if self.world.terrain[x, y] == 2:
      return
    else:
      self.coords = np.array([x, y])

  def _right(self):
    self.heading = (self.heading - 1) % 8

  def _eat(self):
    x, y = self.coords
    food = self.world.food_amount[x, y]
    protein_hue = self.world.protein_hue[x, y]
    compat = 0.5 - np.abs(self.protein_hue - protein_hue)
    delta = compat * food
    self.energy += delta
    self.world.food_amount[x, y] = 0

  def _reproduce(self):
    # Can you birth?
    if self.energy < 2.0:
      return
    anti_heading = (self.heading - 4) % 8
    x, y = self.coords + np.array(self.HEADINGS[anti_heading])
    if not self.world.valid_coords(x, y):
      return
    if self.world.terrain[x, y] == 2:
      return

    # Child is a noisy copy
    child = Beast(self.world)
    child.name = Namer.name()
    child.coords = np.array([x, y])
    child.protein_hue = (self.protein_hue + np.random.normal(0, 0.1)) % 1.0
    child.pelt_hue = (self.pelt_hue + np.random.normal(0, 0.1)) % 1.0
    child.synapses = self.synapses + np.random.normal(
        0, 0.005, size=self.synapses.shape)
    child.brain = Brain(child)
    child.age = 0
    child.generation = self.generation + 1
    
    # Energy divided between parent and child
    self.energy -= self.REPRODUCTION_COST
    self.energy /= 2.0
    child.energy = self.energy
    return child


  def _take_action(self, action_ix):
    if action_ix == 0:
      self._left()
    elif action_ix == 1:
      self._forward()
    elif action_ix == 2:
      self._right()
    elif action_ix == 3:
      self._eat()
    elif action_ix == 4:
      return self._reproduce()

  def tick(self):
    self.age += 1
    starting_energy = self.energy
    self.energy -= self.METABOLIC_COST 
    x, y = self.coords
    if self.world.terrain[x, y] == 0:
      self.energy -= self.DROWNING_COST * (2 ** (self.age / 500))
    percept = self._perceive()
    action = self.brain.think(percept)
    result = self._take_action(action)
    if action != 4:
      self.brain.learn(self.energy - starting_energy)
    return result
