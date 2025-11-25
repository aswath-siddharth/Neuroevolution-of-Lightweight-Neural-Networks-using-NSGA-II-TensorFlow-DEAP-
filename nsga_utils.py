
from deap import base, creator, tools
import random
from model_utils import evaluate

def setup_toolbox(train_X, train_y, test_X, test_y):

    creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0, -1.0))
    creator.create("Individual", list, fitness=creator.FitnessMulti)

    toolbox = base.Toolbox()
    toolbox.register("num_layers", random.randint, 1, 3)
    toolbox.register("units", random.randint, 16, 128)
    toolbox.register("activation", random.randint, 0, 2)

    def create_individual():
        L = toolbox.num_layers()
        indiv = [L] + [toolbox.units() for _ in range(L)] + [toolbox.activation()]
        return creator.Individual(indiv)

    toolbox.register("individual", create_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register(
        "evaluate",
        evaluate,
        train_X=train_X, train_y=train_y,
        test_X=test_X, test_y=test_y
    )

    toolbox.register("mate", tools.cxTwoPoint)

    def mutate(ind):
        if random.random() < 0.5:
            ind[0] = random.randint(1, 3)
        if random.random() < 0.5:
            for i in range(1, ind[0]+1):
                ind[i] = random.randint(16, 128)
        if random.random() < 0.5:
            ind[-1] = random.randint(0, 2)
        return ind,

    toolbox.register("mutate", mutate)
    toolbox.register("select", tools.selNSGA2)

    return toolbox
