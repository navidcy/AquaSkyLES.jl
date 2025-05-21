using AquaSkyLES.AtmosphereModels: AtmosphereModel
using Oceananigans

grid = RectilinearGrid(size=(16, 16, 16), x=(0, 1), y=(0, 1), z=(0, 1))
model = AtmosphereModel(grid)
