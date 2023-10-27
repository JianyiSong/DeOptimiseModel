import numpy as np
import geatpy as ea
from keras.models import load_model
import tensorflow as tf
import matplotlib.pylab as plt



class MyProblem(ea.Problem):  # Inherit the Problem parent class
    def __init__(self, data, initpars):
        name = 'ZDT1'  # Initialize name (function name, feel free to set it)
        M = 1  # Initialize M (target dimension)
        maxormins = [1] * M  # Initialize maxormins (list of target min-max markers, 1: minimize this target; -1: maximize this target)
        Dim = len(initpars)  # Initialize Dim (Decision Variable Dimension)
        varTypes = [0] * Dim  # Initialize varTypes (type of decision variable, 0: real; 1: integer)
        lb = []  # Lower bounds on decision variables
        ub = []  # Upper bounds on decision variables
        for i in range(Dim):
            lb.append(initpars[i][0])
            ub.append(initpars[i][1])
        lbin = [1] * Dim
        ubin = [1] * Dim
        self.initpars = initpars
        self.phase = data[:, 0]
        self.datay = data[:, 1] - np.mean(data[:, 1])
        self.cost = []

        self.xspace = np.linspace(0, 1, 100)
        self.sigma = np.diff(self.datay, 2).std() / np.sqrt(6)  # Estimated observation noise values
        if Dim == 7:
            self.model = load_model('model10mc.hdf5')
        elif Dim == 8:
            self.model = load_model('model10l3mc.hdf5')
        else:
            raise RuntimeError('Input init-parameters should be with length of 6 or 7.')
        # results
        self.DE_trace = []

        # Instantiation is accomplished by calling the parent class constructor
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)

    def predict(self, allpara):
        arraymcn = np.array(allpara)
        if len(arraymcn) == 7:
            arraymc = arraymcn[0:5]
            mcinput = np.reshape(arraymc, (1, 5))
            lightdata = self.model(mcinput)
            proto_tensor = tf.make_tensor_proto(lightdata)
            data_numpy = tf.make_ndarray(proto_tensor)
            return data_numpy[0] + arraymcn[6]
        else:
            arraymc = arraymcn[0:6]
            mcinput = np.reshape(arraymc, (1, 6))
            lightdata = self.model(mcinput)
            proto_tensor = tf.make_tensor_proto(lightdata)
            data_numpy = tf.make_ndarray(proto_tensor)
            return data_numpy[0] + arraymcn[7]

    def getdata(self, allpara):
        arraymc = np.array(allpara)

        if len(arraymc) == 7:
            _offset = int(arraymc[5])
        else:
            _offset = int(arraymc[6])
        datay_m = np.hstack((self.datay[_offset:], self.datay[:_offset]))
        noise_y = np.interp(self.xspace, self.phase, datay_m)  # y轴
        return noise_y

    def get_lcfit(self, phen):
        return self.predict(phen)

    def aimFunc(self, pop):  # objective function
        cost = []
        x = pop.Chrom
        self.DE_trace.append(x)
        for i in range(pop.sizes):
            output = self.predict(x[i])
            noisey = self.getdata(x[i])
            # Calculating the likelihood function
            cost.append(0.5 * np.sum(np.log(2 * np.pi * self.sigma ** 2) + (output - noisey) ** 2 / (self.sigma ** 2)))
        pop.ObjV = np.reshape(np.array(cost), (-1, 1))  # Compute the objective function value to assign to the ObjV property of the pop population object

def geat(data, initpars, MAXGEN, NIND, Encoding):
    problem = MyProblem(data, initpars)  # Generating Problem Objects
    # Construction Algorithms
    Field = ea.crtfld(Encoding, problem.varTypes, problem.ranges, problem.borders)  # Creating a region descriptor
    population = ea.Population(Encoding, Field, NIND)  # Instantiate the population object (at this point the population has not been initialized, just completing the instantiation of the population object)
    myAlgorithm = ea.soea_DE_best_1_bin_templet(problem, population)  # Instantiate an algorithm template object
    myAlgorithm.MAXGEN = MAXGEN  # Maximum number of evolutionary generations
    myAlgorithm.mutOper.F = 0.5  # The parameter F in differential evolution
    myAlgorithm.recOper.XOVR = 0.7  # probability of reorganization
    myAlgorithm.logTras = 1  # Set how many generations to log, if it is set to 0 it means no logging
    myAlgorithm.verbose = False  # Set whether to print out log messages
    myAlgorithm.drawing = 3  # Setting the plotting mode (0: no plotting; 1: plotting the results; 2: animating the process in target space; 3: animating the process in decision space)
    """===========================Calling algorithmic templates for population evolution========================"""
    [BestIndi, population] = myAlgorithm.run()  # Execute the algorithm template to obtain the optimal individuals as well as the final generation population
    phen = BestIndi.Phen[0]

    pre = problem.get_lcfit(phen)

    if phen.shape[0] == 7:
        offset = int(phen[5])
    else:
        offset = int(phen[6])
    plt.figure()
    ax = plt.gca()
    datay = np.hstack((problem.datay[offset:], problem.datay[:offset]))


    ax.plot(problem.phase, problem.datay, '.', c='k')

    ax.plot(problem.xspace, pre, '-r')
    ax.yaxis.set_ticks_position('left')
    ax.invert_yaxis()  # y-axis reversed
    plt.xlabel('phase', fontsize=18)
    plt.ylabel('mag', fontsize=18)
    plt.show()
    return BestIndi,population


def parameters(initpars, MAXGEN, NIND):
    file = 'KIC 6431545.txt'
    data = np.loadtxt(file)
    BestIndi,population = geat(data, initpars, MAXGEN, NIND, Encoding)


if __name__ == '__main__':
    MAXGEN = 100  # Maximum number of evolutionary generations
    NIND = 100  # population size
    Encoding = 'RI'  # coding method

    initpars = [(4655 / 5850 - 0.8, 4655 / 5850 + 0.8),
                (54.40 / 90 - 90 / 90, 54.40 / 90 + 90 / 90),
                (0, 5),
                (0, 1.9),
                (0.1, 1.5),
                (0, 5),  # l3 par
                (-10, 10),
                (-0.1, 0.1)]
    parameters(initpars, MAXGEN, NIND)
