import numpy as np


class World(object):

  FOOD_AMOUNT_EPSILON = 0.001
  PROTEIN_HUE_EPSILON = 0.0001

  def __init__(self, xsize=100, ysize=100, tpoints=30, bpoints=50):
    # Dimensions of the world, a grid of tiles
    self.xsize = xsize
    self.ysize = ysize

    # Terrain types: 0=water, 1=land, 2=boulder
    self.terrain = np.ones((xsize, ysize), dtype=np.int8)

    # Vegetation and food!
    # Land tiles have an amount of food, which tends towards the tile's
    # fertility on each tick. They also have a protein biome; the type
    # of food on the tile tends toward's that biome's hue on each tick.
    #
    # The food level that the tile tends towards.
    self.fertility = np.ones((xsize, ysize), dtype=np.float32)
    # Food currently available.
    self.food_amount = np.zeros_like(self.fertility)
    # Type of food that the tile tends towards.
    self.protein_biome = np.zeros_like(self.fertility)
    # Determines who can eat this tile's food, and who it poisons.
    self.protein_hue = np.zeros_like(self.fertility)

    # Generate some starting values
    self._build_map(tpoints, bpoints)

    # Book keeping
    self.age = 0

  def _attractor_field(self, n_high, n_low):
    # Grids of x and y coords
    yy = np.tile(np.arange(self.ysize), self.xsize).reshape(
        (self.xsize, self.ysize))
    xx = np.repeat(np.arange(self.xsize), self.ysize).reshape(
        (self.xsize, self.ysize))
    xscale = self.xsize / 20.0
    yscale = self.ysize / 20.0

    # Attractor points
    high_xs = np.random.randint(self.xsize, size=n_high)
    high_ys = np.random.randint(self.ysize, size=n_high)
    low_xs = np.random.randint(self.xsize, size=n_low)
    low_ys = np.random.randint(self.ysize, size=n_low)

    # Force of each high attractor
    high_x_dists = [((xx - high_x) / xscale) ** 2 for high_x in high_xs]
    high_y_dists = [((yy - high_y) / yscale) ** 2 for high_y in high_ys]
    high_dists = [hx + hy for hx, hy in zip(high_x_dists, high_y_dists)]
    high_pulls = [1.0 / (1.0 + hd) for hd in high_dists]

    # And each low attractor
    low_x_dists = [((xx - low_x) / xscale) ** 2 for low_x in low_xs]
    low_y_dists = [((yy - low_y) / yscale) ** 2 for low_y in low_ys]
    low_dists = [lx + ly for lx, ly in zip(low_x_dists, low_y_dists)]
    low_pulls = [1.0 / (1.0 + ld) for ld in low_dists]

    return np.tanh(np.sum(high_pulls, axis=0) - np.sum(low_pulls, axis=0))

  def _build_map(self, tpoints, bpoints):
    # Terrain types
    elevation = self._attractor_field(tpoints, tpoints) + np.random.normal(
        0, 0.01, size=(self.xsize, self.ysize))
    water_cutoff = np.percentile(elevation, 40)
    boulder_cutoff = np.percentile(elevation, 95)
    self.terrain[elevation < water_cutoff] = 0
    self.terrain[elevation > boulder_cutoff] = 2

    # Vegetation
    self.fertility = (self._attractor_field(2, 2) + 1.0) * 1.0
    self.food_amount = self.fertility.copy()
    self.protein_biome = (self._attractor_field(bpoints, bpoints) + 1.0) / 2.0
    self.protein_hue = self.protein_biome.copy()

  def tick(self):
    self.food_amount += (self.terrain == 1) * self.FOOD_AMOUNT_EPSILON * (
        self.fertility - self.food_amount)
    self.protein_hue += self.PROTEIN_HUE_EPSILON * (
        self.protein_biome - self.protein_hue)
    self.age += 1

  def valid_coords(self, x, y):
    if x < 0 or y < 0 or x >= self.xsize or y >= self.ysize:
      return False
    return True

  def on_land(self, x, y):
    if not self.valid_coords(x, y):
        return False
    if self.terrain[x, y] == 1:
        return True
    return False

  def percept_at(self, x, y):
    if not self.valid_coords(x, y):
      return 0.0, 0.0, 1.0
    return (
      self.protein_hue[x, y],
      self.food_amount[x, y],
      self.terrain[x, y] / 2.0)
