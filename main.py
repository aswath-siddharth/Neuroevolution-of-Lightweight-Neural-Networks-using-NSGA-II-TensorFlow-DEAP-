
from nsga_utils import setup_toolbox
from deap import tools
import random
import tensorflow as tf

(train_X, train_y), (test_X, test_y) = tf.keras.datasets.mnist.load_data()
train_X, test_X = train_X / 255.0, test_X / 255.0

toolbox = setup_toolbox(train_X, train_y, test_X, test_y)

POP = 6
GEN = 3
pop = toolbox.population(n=POP)

fitnesses = list(map(toolbox.evaluate, pop))
for ind, fit in zip(pop, fitnesses):
    ind.fitness.values = fit

for g in range(GEN):
    print("Generation", g)

    offspring = tools.selTournamentDCD(pop, len(pop))
    offspring = list(map(toolbox.clone, offspring))

    for c1, c2 in zip(offspring[::2], offspring[1::2]):
        if random.random() < 0.9:
            toolbox.mate(c1, c2)
        toolbox.mutate(c1)
        toolbox.mutate(c2)
        del c1.fitness.values
        del c2.fitness.values

    invalids = [ind for ind in offspring if not ind.fitness.valid]
    fits = map(toolbox.evaluate, invalids)
    for ind, fit in zip(invalids, fits):
        ind.fitness.values = fit

    pop = toolbox.select(pop + offspring, POP)

front = tools.sortNondominated(pop, len(pop))[0]
print("Final Pareto Front:", front)
