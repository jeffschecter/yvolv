import numpy as np

from yvolv import beast
from yvolv import world


class Simulation(object):

  def __init__(self, world, n_bsts):
    self.world = world
    self.graveyard = []
    self.herd = [beast.Beast(world) for _ in np.arange(n_bsts)]

  def _survives(self, bst):
    if bst.energy <= 0:
      return False
    x, y = bst.coords
    if self.world.terrain[x, y] == 2:
      return False
    return True

  def tick(self):
    self.graveyard += [
        bst for bst in self.herd if not self._survives(bst)]
    self.herd = [bst for bst in self.herd if self._survives(bst)]
    births = []
    for bst in self.herd:
      result = bst.tick()
      if isinstance(result, beast.Beast):
        births.append(result)
    self.herd += births
    self.world.tick()

  def image(self):
    rgb = np.stack(
      [1 - 0.3 * self.world.protein_hue.T,
       0.9 * np.ones_like(self.world.protein_hue.T),
       1 - 0.3 * (np.abs(self.world.protein_hue.T - 0.5) * 2)],
      axis=-1)
    rgb *= np.expand_dims(0.8 + 0.1 * self.world.food_amount.T, -1)
    rgb[self.world.terrain.T == 0] = [0.6, 0.2, 0.8]
    rgb[self.world.terrain.T == 2] = [0.5, 0.5, 0.5]
    for bst in self.herd:
      x, y = bst.coords
      rgb[y, x, :] = [0, 0, 0]
    return np.flip(rgb, axis=0)