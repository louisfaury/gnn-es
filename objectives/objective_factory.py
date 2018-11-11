
from objectives.objectives import Sphere, Rosenbrock, Rastrigin, Ackley, SixHumpCamel, Styblinski, Cigar


def create_objective(config, dim):
    name = config["name"].lower()
    if name == 'sphere':
        objective = Sphere(dim)
    elif name == 'rosenbrock':
        objective = Rosenbrock(dim)
    elif name == 'rastrigin':
        objective = Rastrigin(dim)
    elif name == 'ackley':
        objective = Ackley(dim)
    elif name == 'sixhumpcamel':
        objective = SixHumpCamel(dim)
    elif name == 'styblinski':
        objective = Styblinski(dim)
    elif name == 'cigar':
        objective = Cigar(dim)
    else:
        raise NotImplementedError('Unknown objective function: ' + str(name))
    return objective
