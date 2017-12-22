import numpy as np

from yvolv import beast
from yvolv import world


class Simulation(object):

  def __init__(self, world, n_bsts):
    self.world = world
    self.herd = [beast.Beast(world) for _ in np.arange(n_bsts)]
    self.herd = [
        bst for bst in self.herd
        if self.world.on_land(*list(bst.coords))]

    # Book keeping
    self.births = []
    self.deaths = []
    self.population = []
    self.pop_nw = []
    self.pop_ne = []
    self.pop_sw = []
    self.pop_se = []
    self.red_eaters = []
    self.blue_eaters = []
    self.green_eaters = []
    self.food_amount = []
    self.red_food = []
    self.blue_food = []
    self.green_food = []
    self.herd_age = []
    self.herd_energy = []
    self.herd_generation = []
    self.lefts = []
    self.forwards = []
    self.rights = []
    self.eats = []
    self.reproduces = []

  def _survives(self, bst):
    if bst.energy <= 0:
      return False
    x, y = bst.coords
    if self.world.terrain[x, y] == 2:
      return False
    return True

  def tick(self):
    survivors = []
    deaths = []
    births = []
    actions = {}

    # Prune the dead
    for bst in self.herd:
      if self._survives(bst):
        survivors.append(bst)
      else:
        deaths.append(bst)
    self.herd = survivors

    # Tick each beast, add births to herd
    for bst in self.herd:
      action, result = bst.tick()
      actions[action] = actions.get(action, 0) + 1
      if isinstance(result, beast.Beast):
        births.append(result)
    self.herd += births

    # Tick the world
    self.world.tick()

    # Update the book
    ns = self.world.ysize / 2.0
    we = self.world.xsize / 2.0
    self.births.append(len(births))
    self.deaths.append(len(deaths))
    self.population.append(len(self.herd))
    self.pop_nw.append(sum(
        1 for bst in self.herd
        if bst.coords[0] < we and bst.coords[1] > ns))
    self.pop_ne.append(sum(
        1 for bst in self.herd
        if bst.coords[0] > we and bst.coords[1] > ns))
    self.pop_sw.append(sum(
        1 for bst in self.herd
        if bst.coords[0] > we and bst.coords[1] < ns))
    self.pop_se.append(sum(
        1 for bst in self.herd
        if bst.coords[0] < we and bst.coords[1] < ns))
    self.red_eaters.append(sum(
        1 for bst in self.herd if bst.protein_hue > 0.66))
    self.blue_eaters.append(sum(
        1 for bst in self.herd
        if bst.protein_hue <= 0.66 and bst.protein_hue >= 0.33))
    self.green_eaters.append(sum(
        1 for bst in self.herd if bst.protein_hue < 0.33))
    self.food_amount.append(self.world.food_amount.sum())
    self.red_food.append(self.world.food_amount[
        self.world.protein_hue > 0.66].sum())
    self.blue_food.append(self.world.food_amount[
        (self.world.protein_hue <= 0.66) &
        (self.world.protein_hue >= 0.33)].sum())
    self.green_food.append(self.world.food_amount[
        self.world.protein_hue < 0.33].sum())
    self.herd_age.append(np.mean([bst.age for bst in self.herd]))
    self.herd_energy.append(np.mean([bst.energy for bst in self.herd]))
    self.herd_generation.append(np.mean([bst.generation for bst in self.herd]))
    self.lefts.append(actions.get(0, 0))
    self.forwards.append(actions.get(1, 0))
    self.rights.append(actions.get(2, 0))
    self.eats.append(actions.get(3, 0))
    self.reproduces.append(actions.get(4, 0))

  def image(self):
    rgb = np.stack(
      [1 - 0.3 * self.world.protein_hue.T,
       0.9 * np.ones_like(self.world.protein_hue.T),
       1 - 0.3 * (np.abs(self.world.protein_hue.T - 0.5) * 2)],
      axis=-1)
    rgb *= np.minimum(
        1.0, np.expand_dims(0.7 + 2.0 * self.world.food_amount.T, -1))
    rgb[self.world.terrain.T == 0] = [0.6, 0.2, 1.0]
    rgb[self.world.terrain.T == 2] = [0.3, 0.3, 0.3]
    for bst in self.herd:
      x, y = bst.coords
      rgb[y, x, :] = bst.pelt_rgb
    return np.flip(rgb, axis=0)
