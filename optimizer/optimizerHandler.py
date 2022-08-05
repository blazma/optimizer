from fitnessFunctions import fF,frange
from optionHandler import optionHandler
#import Core
import sys
import logging
import numpy as np
import copy
import random
import json
import time
import os
from math import sqrt

from multiprocessing import Pool

from itertools import combinations, product

from types import MethodType
try:
    import copyreg
except:
    import copyreg

import functools
try:
    import cPickle as pickle
except ImportError:
    import pickle
				

def _pickle_method(method):
	func_name = method.__func__.__name__
	obj = method.__self__
	cls = method.__self__.__class__
	return _unpickle_method, (func_name, obj, cls)

def _unpickle_method(func_name, obj, cls):
	for cls in cls.mro():
		try:
			func = cls.__dict__[func_name]
		except KeyError:
			pass
		else:
			break
	return func.__get__(obj, cls)


try:
	copyreg.pickle(MethodType, _pickle_method, _unpickle_method)
except:
	copyreg.pickle(MethodType, _pickle_method, _unpickle_method)


def normalize(values,args):
	"""
	Normalizes the values of the given ``list`` using the defined boundaries.

	:param v: the ``list`` of values
	:param args: an object which has a ``min_max`` attribute which consists of two ``lists``
		each with the same number of values as the given list

	:return: the ``list`` of normalized values

	"""
	copied_values = copy.copy(values)
	for i in range(len(values)):
		copied_values[i]=(values[i]-args.min_max[0][i])/(args.min_max[1][i]-args.min_max[0][i])
	return copied_values


def uniform(random,args):
	"""
	Creates random values from a uniform distribution. Used to create initial population.

	:param random: random number generator object
	:param args: ``dictionary``, must contain key "num_params" and either "_ec" or "self"

	:return: the created random values in a ``list``

	"""
	size=args.get("num_params")
	bounds=args.get("_ec",args.get("self")).bounder
	candidate=[]
	for i in range(int(size)):
		candidate.append(random.uniform(bounds.lower_bound[i],bounds.upper_bound[i]))
	return candidate

def uniformz(random,size,bounds):
	"""
	Creates random values from a uniform distribution. Used to create initial population.

	:param random: random number generator object
	:param args: ``dictionary``, must contain key "num_params" and either "_ec" or "self"

	:return: the created random values in a ``list``

	"""
	candidate=[]
	for i in range(int(size)):
		candidate.append(random.uniform(bounds.lower_bound[i],bounds.upper_bound[i]))
	return candidate


class my_candidate():
	"""
	Mimics the behavior of ``candidate`` from the ``inspyred`` package to allow the uniform
	handling of the results produced by the different algorithms.

	:param vals: the result of the optimization
	:param fitn: the fitness of the result

	"""
	def __init__(self,vals, fitn=-1):
		self.candidate=vals
		self.fitness=fitn

class SINGLERUN():
	"""
	An abstract base class to implement an optimization process.
	"""
	def __init__(self, reader_obj, option_obj):
		self.fit_obj = fF(reader_obj,  option_obj)
		self.SetFFun(option_obj)
		self.directory = option_obj.base_dir
		self.num_params = option_obj.num_params

	def SetFFun(self,option_obj):
		"""
		Sets the combination function and converts the name of the fitness functions into function instances.

		:param option_obj: an ``optionHandler`` instance

		"""

		try:
			self.ffun=self.fit_obj.fun_dict["single_objective"]
		except KeyError:
			sys.exit("Unknown fitness function!")

		if option_obj.type[-1] != 'features':
			try:
				option_obj.feats=[self.fit_obj.calc_dict[x] for x in option_obj.feats]
			except KeyError:
				print("error with fitness function: ",option_obj.feats," not in: ",list(self.fit_obj.calc_dict.keys()))

	

class oldBaseOptimizer():
	"""
	An abstract base class to implement an optimization process.
	"""
	def __init__(self):
		pass
	def SetFFun(self,option_obj):
		"""
		Sets the combination function and converts the name of the fitness functions into function instances.

		:param option_obj: an ``optionHandler`` instance

		"""

		try:
			self.ffun = self.fit_obj.fun_dict["single_objective"]
			self.mfun = self.fit_obj.fun_dict["multi_objective"]
		except KeyError:
			sys.exit("Unknown fitness function!")

		if option_obj.type[-1]!= 'features':
			try:
				option_obj.feats = [self.fit_obj.calc_dict[x] for x in option_obj.feats]
			except KeyError:
				print("error with fitness function: ",option_obj.feats," not in: ",list(self.fit_obj.calc_dict.keys()))



# to generate a new set of parameters
class baseOptimizer():
	"""
	An abstract base class to implement an optimization process.
	"""
	def __init__(self, reader_obj,  option_obj):
		self.fit_obj = fF(reader_obj,  option_obj)
		self.SetFFun(option_obj)
		self.rand = random
		self.seed = int(option_obj.seed)
		self.rand.seed(self.seed)
		self.directory = option_obj.base_dir
		self.num_params = option_obj.num_params
		if option_obj.type[-1]!= "features":
			self.number_of_traces = reader_obj.number_of_traces()
		else:
			self.number_of_traces = len(reader_obj.features_data["stim_amp"])
		self.num_obj = self.num_params*int(self.number_of_traces)
		self.min_max = option_obj.boundaries

	def SetFFun(self,option_obj):
		"""
		Sets the combination function and converts the name of the fitness functions into function instances.

		:param option_obj: an ``optionHandler`` instance

		"""

		try:
			self.ffun = self.fit_obj.fun_dict["single_objective"]
			self.mfun = self.fit_obj.fun_dict["multi_objective"]
		except KeyError:
			sys.exit("Unknown fitness function!")
		
		if option_obj.type[-1]!= 'features':
			try:
				option_obj.feats = [self.fit_obj.calc_dict[x] for x in option_obj.feats]
			except KeyError:
				print("error with fitness function: ",option_obj.feats," not in: ",list(self.fit_obj.calc_dict.keys()))

	

class InspyredAlgorithmBasis(baseOptimizer):
	def __init__(self, reader_obj,  option_obj):
		baseOptimizer.__init__(self, reader_obj,  option_obj)
		import inspyred
		self.inspyred = inspyred
		self.bounder = self.inspyred.ec.Bounder([0]*len(option_obj.boundaries[0]),[1]*len(option_obj.boundaries[1]))
		self.algo_params = copy.copy(option_obj.algorithm_parameters)
		self.size_of_population = self.algo_params.pop("size_of_population")
		self.number_of_generations = self.algo_params.pop("number_of_generations")
		self.stat_file = open(self.directory + "/stat_file.txt", "w")
		self.ind_file = open(self.directory + "/ind_file.txt", "w")
		self.number_of_cpu = int(self.algo_params.pop("number_of_cpu",1))
		"""try:
			# print type(option_obj.starting_points)
			if isinstance(option_obj.starting_points[0], list):
				self.starting_points = option_obj.starting_points
			else:
				self.starting_points = [normalize(option_obj.starting_points, self)]
		except TypeError:"""
		self.starting_points = None
		if option_obj.output_level == "1":
			print("starting points: ", self.starting_points)
		self.kwargs = dict(generator=uniform,
						   evaluator=self.inspyred.ec.evaluators.parallel_evaluation_mp,
						   mp_evaluator=self.ffun,
						   mp_nprocs=int(self.number_of_cpu),
						   pop_size=self.size_of_population,
						   seeds=self.starting_points,
						   max_generations=self.number_of_generations,
						   num_params=self.num_params,
						   maximize=False,
						   bounder=self.bounder,
						   boundaries=self.min_max,
						   statistics_file=self.stat_file,
						   individuals_file=self.ind_file)

				
	def Optimize(self):
			"""
			Performs the optimization.
			"""
			
			logger = logging.getLogger('inspyred.ec')
			logger.setLevel(logging.DEBUG)
			file_handler = logging.FileHandler(self.directory + '/inspyred.log', mode='w')
			file_handler.setLevel(logging.DEBUG)
			formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
			file_handler.setFormatter(formatter)
			logger.addHandler(file_handler)
			self.evo_strat.terminator=self.inspyred.ec.terminators.generation_termination
			self.kwargs={k: v for k, v in self.kwargs.items() if v!='None' and v!=None} #maximize equals none can't use if v
			self.solutions = self.evo_strat.evolve(**self.kwargs, **self.algo_params)
			

			if hasattr(self.evo_strat, "archive"):
				self.final_archive = self.evo_strat.archive
	

class ScipyAlgorithmBasis(baseOptimizer):

	def __init__(self, reader_obj,  option_obj):
		baseOptimizer.__init__(self, reader_obj,  option_obj)

		"""try:
			if isinstance(option_obj.starting_points[0], list):
				raise TypeError
			else:
				self.starting_points = [normalize(option_obj.starting_points, self)]
		except TypeError:"""
		self.starting_points = uniform(self.rand, {"num_params": self.num_params, "self": self})
		if option_obj.output_level == "1":
			print("starting points: ", self.starting_points)

	def wrapper(self, candidates, args):
		"""
		Converts the ``ndarray`` object into a ``list`` and passes it to the fitness function.

		:param candidates: the ``ndarray`` object
		:param args: optional parameters to be passed to the fitness function

		:return: the return value of the fitness function

		"""
		tmp = ndarray.tolist(candidates)
		candidates = self.bounder(tmp, args)
		return self.ffun([candidates], args)[0]

class CMAES_CMAES(baseOptimizer):
	def __init__(self, reader_obj,  option_obj):
		baseOptimizer.__init__(self, reader_obj,  option_obj)
		self.algo_params = copy.copy(option_obj.algorithm_parameters)
		self.size_of_population = self.algo_params.pop("size_of_population")
		self.number_of_generations = self.algo_params.pop("number_of_generations")
		self.number_of_cpu = int(self.algo_params.pop("number_of_cpu",1))
		self.stat_file = open(self.directory + "/stat_file.txt", "w")
		self.ind_file = open(self.directory + "/ind_file.txt", "w")
		self.solutions = []
		"""try:
			if isinstance(option_obj.starting_points[0], list):
				self.starting_points = option_obj.starting_points
			else:
				self.starting_points = [normalize(option_obj.starting_points, self)]
		except TypeError:
			self.starting_points = None"""
		if option_obj.output_level == "1":
			print("starting points: ", self.starting_points)
		from cmaes import CMA
		self.cmaoptimizer = CMA(mean=(np.ones(len(self.min_max[0]))*0.5), **self.algo_params, seed=1234, population_size=int(self.size_of_population), bounds=np.array([[0,1]]*len(self.min_max[0])))
		
				
	def Optimize(self):
			"""
			Performs the optimization.
			"""
			with Pool(int(self.number_of_cpu)) as pool:
				for generation in range(int(self.number_of_generations)):
					print("Generation: {0}".format(generation+1))
					solutions = []
					candidate = [[self.cmaoptimizer.ask()] for _ in range(self.cmaoptimizer.population_size)]
					fitness = pool.map(self.ffun,candidate)
					solutions=[(pop[0], fit[0]) for pop,fit in zip(candidate,fitness)]
					self.cmaoptimizer.tell(solutions)
					[self.solutions.append(my_candidate(c[0],f)) for c,f in zip(candidate,fitness)]
			
			

class PygmoAlgorithmBasis(baseOptimizer):

	def __init__(self, reader_obj,  option_obj):
		baseOptimizer.__init__(self, reader_obj,  option_obj)
		import pygmo as pg
		self.pg = pg
		self.algo_params = copy.copy(option_obj.algorithm_parameters)
		self.number_of_generations = int(self.algo_params.pop("number_of_generations"))
		self.size_of_population = int(self.algo_params.pop("size_of_population"))
		self.multiobjective=False
		self.multiprocessing=False
		self.option_obj=option_obj
		self.pg.set_global_rng_seed(seed = self.seed)
		self.boundaries = option_obj.boundaries
		self.base_dir = option_obj.base_dir
		if self.option_obj.type[-1]!="features":
			self.number_of_traces=reader_obj.number_of_traces()
		else:
			self.number_of_traces=len(reader_obj.features_data["stim_amp"])
		self.n_obj=len(option_obj.GetFitnessParam()[-1])*int(self.number_of_traces)
		self.number_of_cpu = int(self.algo_params.pop("number_of_cpu",1))
		self.num_islands = int(self.algo_params.pop("number_of_islands",1))

	def Optimize(self):
		
		if self.multiobjective:
			fitfun=self.mfun
		else:
			fitfun=self.ffun
			self.n_obj=1
		self.prob = Problem(fitfun,self.boundaries, self.num_islands, self.size_of_population,1,self.n_obj,self.base_dir)		
		
		if self.multiprocessing:
			self.mpbfe=self.pg.mp_bfe()
			self.mpbfe.resize_pool(int(self.number_of_cpu))
			self.algorithm.set_bfe(self.pg.bfe())
			self.pgalgo=self.pg.algorithm(self.algorithm)
			self.pgalgo.set_verbosity(1)
			self.archi = self.pg.population(prob=self.prob, size=self.size_of_population,b=self.mpbfe)
			self.archi = self.pgalgo.evolve(self.archi)
			
			self.mpbfe.shutdown_pool()
			
			if self.multiobjective:
				self.champions_x = self.archi.get_x()
				self.champions_f = list(np.mean(self.archi.get_f(),axis=1))
				self.best_fitness = min(self.champions_f)
				self.best = self.best = normalize(self.champions_x[self.champions_f.index(self.best_fitness)], self)
			else:
				self.champions_x = self.archi.champion_x
				self.champions_f = self.archi.champion_f
				self.best_fitness = self.champions_f
				self.best = normalize(self.champions_x, self)
					
		else:
			self.pgalgo=self.pgalgorithm(self.algorithm)
			self.pgalgo.set_verbosity(1)
			self.archi = self.pg.archipelago(n=self.num_islands,t = self.pg.fully_connected(),algo=self.pgalgo, prob=self.prob, pop_size=self.size_of_population)#,b=self.bfe)
			self.archi.evolve()
			self.archi.wait()
			for x in range(self.num_islands):
				self.archi.push_back()
		
			champions_x = self.archi.get_champions_x()
			champions_f = self.archi.get_champions_f()
		self.solutions=[my_candidate(c,f) for c,f in zip(champions_x,champions_f)]
			

		
class Problem:
	
	def __init__(self, fitnes_fun, bounds, num_islands=1, size_of_population=1, number_of_generationss=1,n_obj=1, directory=''):
		self.min_max = bounds
		self.fitnes_fun = fitnes_fun
		self.num_islands = num_islands
		self.size_of_population = size_of_population
		self.number_of_generationss = number_of_generationss
		self.pop_counter, self.gen_counter  = 0, 0
		self.directory = directory
		self.nobj=n_obj
		

	def fitness(self, x):
		if self.nobj!=1:
			fitness = self.fitnes_fun([normalize(x,self)])[0]
		else:
			fitness = self.fitnes_fun([normalize(x,self)])
		return fitness

	
	def get_nobj(self):
		return self.nobj

	def get_bounds(self):
		return(self.min_max[0], self.min_max[1])


class SinglePygmoAlgorithmBasis(baseOptimizer):

	def __init__(self, reader_obj,  option_obj):
		baseOptimizer.__init__(self, reader_obj,  option_obj)
		import pygmo as pg
		self.pg = pg
		self.pgset_global_rng_seed(seed = self.seed)
		self.prob = SingleProblem(self.ffun,option_obj.boundaries)
		self.directory = option_obj.base_dir

		self.pop_kwargs = dict()

	def Optimize(self):
		self.population = self.pg.population(self.prob, **self.pop_kwargs)

		self.algorithm.set_verbosity(1)
		self.evolved_pop = self.algorithm.evolve(self.population)
		
		uda = self.algorithm.extract(self.algo_type)
		self.log = uda.get_log()
		self.write_statistics_file()

		self.best = normalize(self.evolved_pop.champion_x, self) 
		self.best_fitness = self.evolved_pop.champion_f

class BluepyoptAlgorithmBasis(baseOptimizer):
	def __init__(self, reader_obj,  option_obj):
		baseOptimizer.__init__(self, reader_obj,  option_obj)
		import bluepyopt as bpop
		self.bpop = bpop
		self.fit_obj = fF(reader_obj,option_obj)
		self.option_obj = option_obj
		self.seed = option_obj.seed
		self.selector_name  =  "IBEA"
		self.directory = str(option_obj.base_dir)
		self.algo_params = copy.copy(option_obj.algorithm_parameters)
		self.size_of_population = self.algo_params.pop("size_of_population")
		self.number_of_generations = self.algo_params.pop("number_of_generations")
		self.num_params = option_obj.num_params
		self.number_of_cpu = int(self.algo_params.pop("number_of_cpu",1))
		self.min_max = option_obj.boundaries
		self.param_names = self.option_obj.GetObjTOOpt()
		self.solutions = []
		if self.option_obj.type[-1]!= "features":
			self.number_of_traces = reader_obj.number_of_traces()
		else:
			self.number_of_traces = len(reader_obj.features_data["stim_amp"])

	import bluepyopt as bpop
	def Optimize(self):
		if self.number_of_cpu > 1:
			from ipyparallel import Client
			print("******************PARALLEL RUN : " + self.selector_name + " *******************")
			os.system("ipcluster start -n "+str(int(self.number_of_cpu))+"  &")
			c = Client(timeout = 60)
			view = c.load_balanced_view()
			view.map_sync(os.chdir, [str(os.path.dirname(os.path.realpath(__file__)))]*int(self.number_of_cpu))
			map_function = view.map_sync
			optimisation = self.bpop.optimisations.DEAPOptimisation(evaluator = self.DeapEvaluator(self))
			self.solutions, self.hall_of_fame, self.logs, self.hist = optimisation.run(int(self.number_of_generations),cp_frequency = int(self.number_of_generations))
			os.system("ipcluster stop")
		else:
			print("*****************Single Run : " + self.selector_name + " *******************")
			optimisation = self.bpop.optimisations.DEAPOptimisation(evaluator = self.DeapEvaluator(self))
			self.solution, self.hall_of_fame, self.logs, self.hist = optimisation.run(int(self.number_of_generations))
	
	class DeapEvaluator(bpop.evaluators.Evaluator):
		def __init__(self,parent):
			super(self.__class__,self).__init__()
			self.parent = parent
			feats = list(zip(self.parent.option_obj.feat_str.split(', '),self.parent.option_obj.weights))
			self.params = [self.parent.bpop.parameters.Parameter(p_name, bounds=(0,1)) for p_name in self.parent.param_names]
			self.param_names = [param.name for param in self.params]
			self.objectives = [self.parent.bpop.objectives.Objective(name=name, value=value) for name,value in feats*self.parent.number_of_traces]


		def evaluate_with_lists(self, param_values):
			err=self.parent.ffun([param_values])
			self.parent.solutions.append(my_candidate(param_values,err[0]))
			return err


class SingleProblem:
	
	def __init__(self, fitnes_fun, bounds):
		self.bounds = bounds
		self.min_max = bounds
		self.fitnes_fun = fitnes_fun

	def __getstate__(self):
		bounds = self.bounds
		min_max = self.min_max
		f_f = self.fitnes_fun
		return (bounds, min_max, f_f)

	def __setstate__(self, state):
		self.bounds, self.min_max, self.fitnes_fun = state


	def fitness(self, x):
		return self.fitnes_fun([normalize(x,self)])

	def get_bounds(self):
		return(self.bounds[0], self.bounds[1])

class RANDOM_SEARCH(baseOptimizer):
	"""
	Implements the ``Differential Evolution Algorithm`` algorithm for minimization from the ``inspyred`` package.
	:param reader_obj: an instance of ``DATA`` object
	:param model_obj: an instance of a model handler object, either ``externalHandler`` or ``modelHandlerNeuron``
	:param option_obj: an instance of ``optionHandler`` object


	.. seealso::

		Documentation of the options from 'inspyred':
			http://inspyred.github.io/reference.html#module-inspyred.ec


	"""
	def __init__(self, reader_obj,  option_obj):
		baseOptimizer.__init__(self, reader_obj,  option_obj)
		self.algo_params =  copy.copy(option_obj.algorithm_parameters)		
		self.directory = str(option_obj.base_dir)
		self.number_of_cpu = int(self.algo_params.pop("number_of_cpu",1))
		for file_name in ["stat_file.txt", "ind_file.txt"]:
			try:
				os.remove(file_name)
			except OSError:
				pass
				


	def Optimize(self):
		"""
		Performs the optimization.
		"""
		with Pool(processes=int(self.number_of_cpu),maxtasksperchild=1) as pool:
			candidate=[]
			fitness=[]
			for j in range(int(self.algo_params["size_of_population"])):
				candidate.append([uniform(self.rand, {"self":self,"num_params":self.num_params})])
			try:
				fitness=pool.map(self.ffun,candidate)
			except (OSError, RuntimeError) as e:
				raise
		self.solutions=[my_candidate(c[0],f) for c,f in zip(candidate,fitness)]
		



class SDE_PYGMO(SinglePygmoAlgorithmBasis):
	
	def __init__(self, reader_obj,  option_obj):

		SinglePygmoAlgorithmBasis.__init__(self, reader_obj,  option_obj)
		self.algo_params =  copy.copy(option_obj.algorithm_parameters)
		self.pop_kwargs['size'] = int(self.algo_params.pop("size_of_population"))

		self.algo_type = self.pg.de        
		self.algorithm = self.pg.de(gen=self.number_of_generations, **self.algo_params)



class bounderObject(object):  #?!
	"""
	Creates a callable to perform the bounding of the parameters.
	:param xmax: list of maxima
	:param xmin: list of minima
	"""
	def __init__(self, xmax, xmin ):
			self.lower_bound = np.array(xmax)
			self.upper_bound = np.array(xmin)
	def __call__(self, **kwargs):
		"""
		Performs the bounding by deciding if the given point is in the defined region of the parameter space.
		This is required by some algorithms as part of their acceptance tests.

		:return: `True` if the point is inside the given bounds.
		"""
		x = kwargs["x_new"]
		tmax = bool(np.all(x <= self.lower_bound))
		tmin = bool(np.all(x >= self.upper_bound))
		return tmax and tmin




class PRAXIS_PYGMO(PygmoAlgorithmBasis):
	def __init__(self, reader_obj,  option_obj):
		PygmoAlgorithmBasis.__init__(self, reader_obj,  option_obj)
		self.algorithm = self.pg.nlopt(solver="praxis", **self.algo_params)
		self.algorithm.maxeval=int(self.algo_params.pop("number_of_generations"))


class NM_PYGMO(PygmoAlgorithmBasis):
	def __init__(self, reader_obj,  option_obj):
		PygmoAlgorithmBasis.__init__(self, reader_obj,  option_obj)
		self.algorithm = self.pg.scipy_optimize(method="Nelder-Mead", options={'xatol': 0, 'fatol': 0})

class BH_PYGMO(PygmoAlgorithmBasis):
	def __init__(self, reader_obj,  option_obj):
		PygmoAlgorithmBasis.__init__(self, reader_obj,  option_obj)
		self.algorithm = self.pg.mbh(self.pg.algorithm(self.pg.scipy_optimize(method="L-BFGS-B")),stop=2)

class DE_PYGMO(PygmoAlgorithmBasis):
	def __init__(self, reader_obj,  option_obj):
		PygmoAlgorithmBasis.__init__(self, reader_obj,  option_obj)
		self.algorithm = self.pg.de(gen=self.number_of_generations, **self.algo_params)

class CMAES_PYGMO(PygmoAlgorithmBasis):
	def __init__(self, reader_obj,  option_obj):
		PygmoAlgorithmBasis.__init__(self, reader_obj,  option_obj)
		
		if int(self.size_of_population) < 5:
			print("***************************************************")
			print("CMA-ES NEEDS A POPULATION WITH AT LEAST 5 INDIVIDUALS")
			print("***************************************************")
			self.size_of_population = 5
		self.algorithm = self.pg.cmaes(gen=self.number_of_generations, **self.algo_params)
		
class PSO_PYGMO(PygmoAlgorithmBasis):
	def __init__(self, reader_obj,  option_obj):
		PygmoAlgorithmBasis.__init__(self, reader_obj,  option_obj)
		self.algorithm = self.pg.pso(gen=self.number_of_generations, **self.algo_params)

class PSOG_PYGMO(PygmoAlgorithmBasis):
	def __init__(self, reader_obj,  option_obj):
		PygmoAlgorithmBasis.__init__(self, reader_obj,  option_obj)
		
		self.multiprocessing=True
		self.algorithm = self.pg.pso_gen(gen=self.number_of_generations, **self.algo_params)

class MACO_PYGMO(PygmoAlgorithmBasis):
	def __init__(self, reader_obj,  option_obj):
		PygmoAlgorithmBasis.__init__(self, reader_obj,  option_obj)
		
		self.multiobjective=True
		self.multiprocessing=True
		self.algorithm = self.pg.maco(gen=self.number_of_generations, **self.algo_params)

class GACO_PYGMO(PygmoAlgorithmBasis):
	def __init__(self, reader_obj,  option_obj):
		PygmoAlgorithmBasis.__init__(self, reader_obj,  option_obj)
		
		self.multiprocessing=True
		self.algorithm = self.pg.gaco(gen=self.number_of_generations, **self.algo_params)

class NSPSO_PYGMO(PygmoAlgorithmBasis):
	def __init__(self, reader_obj,  option_obj):
		PygmoAlgorithmBasis.__init__(self, reader_obj,  option_obj)
		
		self.multiobjective=True
		self.multiprocessing=True
		self.algorithm = self.pg.nspso(gen=self.number_of_generations, **self.algo_params)

class NSGA2_PYGMO(PygmoAlgorithmBasis):
	def __init__(self, reader_obj,  option_obj):
		PygmoAlgorithmBasis.__init__(self, reader_obj,  option_obj)
		
		self.multiobjective=True
		self.multiprocessing=True
		self.algorithm = self.pg.nsga2(gen=self.number_of_generations, **self.algo_params)

class XNES_PYGMO(PygmoAlgorithmBasis):
	def __init__(self, reader_obj,  option_obj):
		PygmoAlgorithmBasis.__init__(self, reader_obj,  option_obj)
		
		self.algorithm = self.pg.xnes(gen=self.number_of_generations, **self.algo_params)

class ABC_PYGMO(PygmoAlgorithmBasis):
	def __init__(self, reader_obj,  option_obj):
		PygmoAlgorithmBasis.__init__(self, reader_obj,  option_obj)
		
		self.algorithm = self.pg.bee_colony(gen=self.number_of_generations, **self.algo_params)

class SGA_PYGMO(PygmoAlgorithmBasis):
	def __init__(self, reader_obj,  option_obj):
		PygmoAlgorithmBasis.__init__(self, reader_obj,  option_obj)
		
		self.algorithm = self.pg.sga(gen=self.number_of_generations, **self.algo_params)

class SADE_PYGMO(PygmoAlgorithmBasis):

	def __init__(self, reader_obj,  option_obj):
		PygmoAlgorithmBasis.__init__(self, reader_obj,  option_obj)
		
		if int(self.size_of_population)<7:
			print("***************************************************")
			print("SADE NEEDS A POPULATION WITH AT LEAST 7 INDIVIDUALS")
			print("***************************************************")
			self.size_of_population = 7
			
		self.algorithm = self.pg.sade(gen=self.number_of_generations, **self.algo_params)

class DE1220_PYGMO(PygmoAlgorithmBasis):

	def __init__(self, reader_obj,  option_obj):

		PygmoAlgorithmBasis.__init__(self, reader_obj,  option_obj)

		if int(self.size_of_population)<7:
			print("*****************************************************")
			print("DE1220 NEEDS A POPULATION WITH AT LEAST 7 INDIVIDUALS")
			print("*****************************************************")
			self.size_of_population = 7
			
		self.algorithm = self.pg.de1220(gen=self.number_of_generations, **self.algo_params)



class BH_SCIPY(ScipyAlgorithmBasis):
	"""
	Implements the ``Basinhopping`` algorithm for minimization from the ``scipy`` package.

	:param reader_obj: an instance of ``DATA`` object
	:param model_obj: an instance of a model handler object, either ``externalHandler`` or ``modelHandlerNeuron``
	:param option_obj: an instance of ``optionHandler`` object

	.. seealso::

		Documentation of the Simulated Annealing from 'scipy':
			http://docs.scipy.org/doc/scipy-dev/reference/generated/scipy.optimize.basinhopping.html

	"""
	def __init__(self, reader_obj,  option_obj):
		ScipyAlgorithmBasis.__init__(self, reader_obj,  option_obj)
		self.algo_params =  copy.copy(option_obj.algorithm_parameters)
		self.maxcor=self.algo_params.pop('maxcor')
		self.ftol=self.algo_params.pop('ftol')
		self.gtol=self.algo_params.pop('gtol')
		self.eps=self.algo_params.pop('eps')
		self.maxfun=self.algo_params.pop('maxfun')
		self.maxiter=self.algo_params.pop('maxiter')
		self.maxls=self.algo_params.pop('maxls')


	def wrapper(self,candidates,args):
		"""
		Converts the ``ndarray`` object into a ``list`` and passes it to the fitness function.

		:param candidates: the ``ndarray`` object
		:param args: optional parameters to be passed to the fitness function

		:return: the return value of the fitness function

		"""
		tmp = ndarray.tolist(candidates)
		ec_bounder = ec.Bounder([0]*len(self.min_max[0]),[1]*len(self.min_max[1]))
		candidates = ec_bounder(tmp,args)
		return self.ffun([candidates],args)[0]



	def Optimize(self):
		"""
		Performs the optimization.
		"""
		self.result=optimize.basinhopping(self.wrapper,
						x0=ndarray((self.num_params,),buffer=array(self.starting_points),offset=0,dtype=float),
						minimizer_kwargs={"method":"L-BFGS-B",
										"jac":False,
										"args":[[]],
										"bounds": [(0,1)]*len(self.min_max[0]),
										"options": {'maxcor': self.maxcor, 
													'ftol': self.ftol, 
													'gtol': self.gtol, 
													'eps': self.eps, 
													'maxfun': self.maxfun, 
													'maxiter': self.maxiter, 
													'maxls': self.maxls,
													'iprint': 2}},
						take_step = None,
						accept_test = self.bounder,
						niter_success = None,
						**self.algo_params
						)
		self.starting_points = uniform(self.rand,{"num_params" : self.num_params,"self": self})
		#print self.result.x
		self.solutions = [my_candidate(self.result.x,self.result.fun)]

	


class NM_SCIPY(ScipyAlgorithmBasis):
	"""
	Implements a Nelder-Mead downhill simplex algorithm for minimization from the ``scipy`` package.

	:param reader_obj: an instance of ``DATA`` object
	:param model_obj: an instance of a model handler object, either ``externalHandler`` or ``modelHandlerNeuron``
	:param option_obj: an instance of ``optionHandler`` object

	.. seealso::

		Documentation of the fmin from 'scipy':
			http://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fmin.html#scipy.optimize.fmin

	"""
	def __init__(self, reader_obj,  option_obj):
		ScipyAlgorithmBasis.__init__(self, reader_obj,  option_obj)
		self.fit_obj = fF(reader_obj,option_obj)
		self.SetFFun(option_obj)
		self.algo_params =  copy.copy(option_obj.algorithm_parameters)
		self.rand = random
		self.seed = option_obj.seed
		self.rand.seed(self.seed)
		self.number_of_generations = self.algo_params.pop("number_of_generations")
		self.num_params = option_obj.num_params
		self.bounder = SetBoundaries(option_obj.boundaries)
		try:
			if isinstance(option_obj.starting_points[0],list):
				raise TypeError
			else:
				self.starting_points = [normalize(option_obj.starting_points,self)]
		except TypeError:
			self.starting_points = uniform(self.rand,{"num_params" : self.num_params,"self": self})
		if option_obj.output_level=="1":
			print("starting points: ",self.starting_points)


	def logger(self,x):
		self.log_file.write(str(x))
		self.log_file.write("\n")
		self.log_file.flush()
		
		
		
	def wrapper(self,candidates,args):
		"""
		Converts the ``ndarray`` object into a ``list`` and passes it to the fitness function.

		:param candidates: the ``ndarray`` object
		:param args: optional parameters to be passed to the fitness function

		:return: the return value of the fitness function

		"""
		tmp = ndarray.tolist(candidates)
		ec_bounder = ec.Bounder([0]*len(self.min_max[0]),[1]*len(self.min_max[1]))
		candidates = ec_bounder(tmp,args)
		fit = self.ffun([candidates],args)[0]
		self.logger(fit)
		return fit




	def Optimize(self):
		"""
		Performs the optimization.
		"""
		self.log_file = open(self.directory + "/nelder.log","w")
		
		list_of_results = [0]*int(self.number_of_generations)
		for points in range(int(self.number_of_generations)):
			
			
			list_of_results[points] = optimize.minimize(self.wrapper,x0 = ndarray((self.num_params,),
						buffer = array(self.starting_points),offset = 0,dtype = float),
									  args = ((),),
									  method = "Nelder-Mead",
									  callback = self.logger,
									  options = {"maxiter":self.size_of_pop,
										  "return_all":True} | self.algo_params
									  )
			#self.log_file.write(str(points+1)+" "+str(self.starting_points)+" ("+str(list_of_results)+") \n")
			
			self.starting_points = uniform(self.rand,{"num_params" : self.num_params,"self": self})
		self.stat_file = open(self.directory + "/ind_file.txt","w")
		self.stat_file.write(str(list_of_results))	
		self.log_file.close()
		self.stat_file.close()
		self.result = min(list_of_results,key = lambda x:x.fun)
		#print self.result.x
		self.solutions = [my_candidate(self.result.x,self.result.fun)]

	



class L_BFGS_B_SCIPY(baseOptimizer):
	"""
	Implements L-BFGS-B algorithm for minimization from the ``scipy`` package.

	:param reader_obj: an instance of ``DATA`` object
	:param model_obj: an instance of a model handler object, either ``externalHandler`` or ``modelHandlerNeuron``
	:param option_obj: an instance of ``optionHandler`` object

	.. seealso::

		Documentation of the L-BFGS-B from 'scipy':
			http://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fmin_l_bfgs_b.html#scipy.optimize.fmin_l_bfgs_b

	"""
	def __init__(self, reader_obj,  option_obj):
		self.fit_obj = fF(reader_obj,option_obj)
		self.SetFFun(option_obj)
		self.rand = random
		self.seed = option_obj.seed
		self.rand.seed(self.seed) 
		self.algo_params =  copy.copy(option_obj.algorithm_parameters)
		self.num_params = option_obj.num_params
		self.min_max = option_obj.boundaries
		self.bounder = SetBoundaries(option_obj.boundaries)
		try:
			if isinstance(option_obj.starting_points[0],list):
				raise TypeError
			else:
				self.starting_points = [normalize(option_obj.starting_points,self)]
		except TypeError:
			self.starting_points = uniform(self.rand,{"num_params" : self.num_params,"self": self})

		if option_obj.output_level == "1":
			print("starting points: ",self.starting_points)


	def wrapper(self,candidates,args):
		"""
		Converts the ``ndarray`` object into a ``list`` and passes it to the fitness function.

		:param candidates: the ``ndarray`` object
		:param args: optional parameters to be passed to the fitness function

		:return: the return value of the fitness function

		"""
		tmp = ndarray.tolist(candidates)
		candidates = self.bounder(tmp,args)
		return self.ffun([candidates],args)[0]




	def Optimize(self):
		"""
		Performs the optimization.
		"""
		self.result = optimize.fmin_l_bfgs_b(self.wrapper,
									  x0 = ndarray((self.num_params,),buffer = array(self.starting_points),offset = 0,dtype=float),
									  args=[[]],
									  bounds= [(0,1)]*len(self.min_max[0]),
									  maxiter= self.number_of_generations,
									  fprime= None,
									  approx_grad= True,
									  **self.algo_params,
									  iprint= 2, #>1 creates log file
									  )
		print(self.result[-1]['warnflag'])
		self.solutions=[my_candidate(self.result[0],self.result[1])]





class grid(baseOptimizer):
	"""
	Implements a brute force algorithm for minimization by calculating the function's value
	over the specified grid.

	.. note::
		This algorithm is highly inefficient and should not be used for complete optimization.

	:param reader_obj: an instance of ``DATA`` object
	:param model_obj: an instance of a model handler object, either ``externalHandler`` or ``modelHandlerNeuron``
	:param option_obj: an instance of ``optionHandler`` object
	:param resolution: number of sample points along each dimensions (default: 10)

	"""
	def __init__(self,reader_obj,option_obj,resolution):
		self.fit_obj=fF(reader_obj,option_obj)
		self.SetFFun(option_obj)
		self.num_params=option_obj.num_params
		self.num_points_per_dim=resolution
		self.min_max=option_obj.boundaries
		self.bounder=SetBoundaries(option_obj.boundaries)



	def Optimize(self,optimals):
		"""
		Performs the optimization.
		"""

		self.solutions=[[],[]]
		_o=copy.copy(optimals)
		_o=normalize(_o, self)
		points=[]
		fitness=[]
		tmp1=[]
		tmp2=[]
		for n in range(self.num_params):
			for c in frange(0,1, float(1)/self.num_points_per_dim):
				_o[n]=c
				tmp1.append(self.fit_obj.ReNormalize(_o))
				tmp2.append(self.ffun([_o],{}))
			points.append(tmp1)
			tmp1=[]
			fitness.append(tmp2)
			tmp2=[]
			_o=copy.copy(optimals)
			_o=normalize(_o, self)
		self.solutions[0]=points
		self.solutions[1]=fitness


	


class CES_INSPYRED(InspyredAlgorithmBasis):
	"""
	Implements a custom version of ``Evolution Strategy`` algorithm for minimization from the ``inspyred`` package.
	:param reader_obj: an instance of ``DATA`` object
	:param model_obj: an instance of a model handler object, either ``externalHandler`` or ``modelHandlerNeuron``
	:param option_obj: an instance of ``optionHandler`` object

	.. note::
		The changed parameters compared to the defaults are the following:
			* replacer: genrational_replacement
			* variator: gaussian_mutation, blend_crossover

	.. seealso::

		Documentation of the options from 'inspyred':
			http://inspyred.github.io/reference.html#module-inspyred.ec


	"""
	def __init__(self, reader_obj,  option_obj):
		InspyredAlgorithmBasis.__init__(self, reader_obj,  option_obj)
		self.evo_strat=ec.ES(self.rand)
		self.evo_strat.selector=self.inspyred.ec.selectors.default_selection
		self.evo_strat.replacer=self.inspyred.ec.replacers.generational_replacement
		self.evo_strat.variator=[self.inspyred.variators.gaussian_mutation,
								 self.inspyred.variators.blend_crossover]
		if option_obj.output_level=="1":
			self.evo_strat.observer=[observers.population_observer,observers.file_observer]
		else:
			self.evo_strat.observer=[observers.file_observer]


class DE_INSPYRED(InspyredAlgorithmBasis):
	"""
	Implements the ``Differential Evolution Algorithm`` algorithm for minimization from the ``inspyred`` package.
	:param reader_obj: an instance of ``DATA`` object
	:param model_obj: an instance of a model handler object, either ``externalHandler`` or ``modelHandlerNeuron``
	:param option_obj: an instance of ``optionHandler`` object

	.. seealso::

		Documentation of the options from 'inspyred':
			http://inspyred.github.io/reference.html#module-inspyred.ec


	"""
	def __init__(self, reader_obj,  option_obj):
		InspyredAlgorithmBasis.__init__(self, reader_obj,  option_obj)
		self.evo_strat=ec.DEA(self.rand)
		self.evo_strat.terminator=terminators.generation_termination
		if option_obj.output_level=="1":
			self.evo_strat.observer=[observers.population_observer,observers.file_observer]
		else:
			self.evo_strat.observer=[observers.file_observer]

class PSO_INSPYRED(InspyredAlgorithmBasis):
	"""
	Implements the ``Particle Swarm`` algorithm for minimization from the ``inspyred`` package.

	:param reader_obj: an instance of ``DATA`` object
	:param model_obj: an instance of a model handler object, either ``externalHandler`` or ``modelHandlerNeuron``
	:param option_obj: an instance of ``optionHandler`` object
	.. seealso::

		Documentation of the Particle Swarm from 'inspyred':
			http://pythonhosted.org/inspyred/reference.html

	"""
	def __init__(self, reader_obj,  option_obj):
		InspyredAlgorithmBasis.__init__(self, reader_obj,  option_obj)
		self.evo_strat=self.inspyred.swarm.PSO(self.rand)
		if option_obj.output_level=="1":
			self.evo_strat.observer=[self.inspyred.ec.observers.population_observer,observers.file_observer]
		else:
			self.evo_strat.observer=[self.inspyred.ec.observers.file_observer]


class NSGA2_INSPYRED(InspyredAlgorithmBasis):
	"""
	Implements the ``Non-Dominated Genetic Algorithm`` algorithm for minimization from the ``inspyred`` package.
	:param reader_obj: an instance of ``DATA`` object
	:param model_obj: an instance of a model handler object, either ``externalHandler`` or ``modelHandlerNeuron``
	:param option_obj: an instance of ``optionHandler`` object

	.. note::
		The changed parameters compared to the defaults are the following:
			* replacer: genrational_replacement
			* variator: gaussian_mutation, blend_crossover

	.. seealso::

		Documentation of the options from 'inspyred':
			http://inspyred.github.io/reference.html#module-inspyred.ec


	"""
	def __init__(self, reader_obj,  option_obj):
		InspyredAlgorithmBasis.__init__(self, reader_obj,  option_obj)
		self.kwargs["mp_evaluator"] = self.mfun
		self.kwargs['mutation_rate'] = option_obj.mutation_rate
		self.evo_strat=ec.emo.NSGA2(self.rand)
		self.evo_strat.selector=inspyred.ec.selectors.default_selection
		self.evo_strat.replacer=inspyred.ec.replacers.nsga_replacement

		self.evo_strat.variator=[variators.gaussian_mutation,
								 variators.blend_crossover]
		if option_obj.output_level=="1":
			self.evo_strat.observer=[observers.population_observer,observers.file_observer]
		else:
			self.evo_strat.observer=[observers.file_observer]



class PAES_INSPYRED(InspyredAlgorithmBasis):
	"""
	Implements a custom version of ``Pareto Archived Evolution Strategies`` algorithm for minimization from the ``inspyred`` package.
	:param reader_obj: an instance of ``DATA`` object
	:param model_obj: an instance of a model handler object, either ``externalHandler`` or ``modelHandlerNeuron``
	:param option_obj: an instance of ``optionHandler`` object


	.. seealso::

		Documentation of the options from 'inspyred':
			http://inspyred.github.io/reference.html#module-inspyred.ec


	"""
	def __init__(self, reader_obj,  option_obj):
		InspyredAlgorithmBasis.__init__(self, reader_obj,  option_obj)

		self.kwargs["mp_evaluator"] = self.mfun


		self.evo_strat=ec.emo.PAES(self.rand)
		self.evo_strat.terminator=terminators.generation_termination
		self.evo_strat.selector=inspyred.ec.selectors.default_selection
		self.evo_strat.replacer=inspyred.ec.replacers.paes_replacement
		self.evo_strat.variator=[variators.gaussian_mutation,
								 variators.blend_crossover]
		if option_obj.output_level=="1":
			self.evo_strat.observer=[observers.population_observer,observers.file_observer]
		else:
			self.evo_strat.observer=[observers.file_observer]

		self.kwargs['mutation_rate'] = option_obj.mutation_rate
		#self.kwargs['num_elites'] = int(4)

class SA_INSPYRED(InspyredAlgorithmBasis):
	"""
	Implements the ``Simulated Annealing`` algorithm for minimization from the ``inspyred`` package.

	:param reader_obj: an instance of ``DATA`` object
	:param model_obj: an instance of a model handler object, either ``externalHandler`` or ``modelHandlerNeuron``
	:param option_obj: an instance of ``optionHandler`` object

	.. seealso::

		Documentation of the Simulated Annealing from 'inspyred':
			http://inspyred.github.io/reference.html#replacers-survivor-replacement-methods


	"""
	def __init__(self, reader_obj,  option_obj):
		InspyredAlgorithmBasis.__init__(self, reader_obj,  option_obj)
		self.kwargs['number_of_generationss'] = self.number_of_generations

		self.evo_strat=self.inspyred.ec.SA(self.rand)
		if option_obj.output_level=="1":
			self.evo_strat.observer=[observers.population_observer,observers.file_observer]
		else:
			self.evo_strat.observer=[observers.file_observer]


class FULLGRID_PYGMO(InspyredAlgorithmBasis):
	
	def __init__(self, reader_obj,  option_obj):
		InspyredAlgorithmBasis.__init__(self, reader_obj,  option_obj)

		self.evo_strat=ec.ES(self.rand)

		if option_obj.output_level=="1":
			self.evo_strat.observer=[observers.population_observer,observers.file_observer]
		else:
			self.evo_strat.observer=[observers.file_observer]
		

		self.resolution = [5,5,5]
		#self.resolution = list(map(lambda x: x if x>=3 else 3, self.resolution))
		
		if(len(self.resolution) < self.kwargs['num_params']):
			print("Not enough values for every parameter. Will expand resolution with threes.")
			self.resolution = self.resolution + [1] * (self.kwargs['num_params'] - len(self.resolution))
			print("New resolution is: ", self.resolution)
			

		elif(len(self.resolution) > self.kwargs['num_params']):
			print("Too many values. Excess resolution will be ignored.")
			self.resolution = self.resolution[0:self.kwargs['num_params']]
			print("New resolution is: ", self.resolution)
			
		
		
		self.grid = [] 
		self.alldims = []
	#self.point = option_obj.point
		#HH
		self.point = [0.12,0.036,0.0003]
		if(not self.point):
			print("No point given. Will take center of grid")
			self.point = list(map(lambda x: int(x/2), self.resolution))
			print("New point is: ", self.point)
			
		#CLAMP
		#self.point = [0.01, 2, 0.3, 3] 
		#align grid on point
		
		for j in range(len(option_obj.boundaries[0])):
			if(self.resolution[j] == 1):
				self.alldims.append([self.point[j]])
				continue
			
			#ugly way to ensure same resolution before and after point included
			upper_bound = option_obj.boundaries[1][j] - float((float(option_obj.boundaries[1][j]))/float(self.resolution[j]-1)/2)
			
			div = float((upper_bound)/(self.resolution[j]-1))
			lower_bound = (self.point[j]/div % 1) * div
			
			upper_bound = upper_bound + lower_bound
			
			self.alldims.append(list(np.linspace(lower_bound,upper_bound,self.resolution[j])))
			
			
		
		for i,t in enumerate(combinations(self.alldims, r=self.num_params-1)):
			plane_dimensions = list(t) 
			optimum_point = [self.point[self.num_params-1-i]]
			plane_dimensions.insert(self.num_params-1-i, optimum_point) 
			print("PLANE", plane_dimensions)

			for t in product(*plane_dimensions):
				print(list(t))
				#exit()
				if(len(self.point)-1-i)==0:
					print(list(t))
					#exit()
				print(list(t))
				self.grid.append(normalize(list(t),self))

				
			if(len(self.point)-1-i)==0:
				print(len(plane_dimensions[0]))



		self.kwargs["seeds"] = self.grid
		self.kwargs["max_generations"] = 0
		self.kwargs["pop_size"] = 1
		

class IBEA_BLUEPYOPT(BluepyoptAlgorithmBasis):
	def __init__(self, reader_obj,  option_obj):
		BluepyoptAlgorithmBasis.__init__(self, reader_obj,  option_obj)
		self.selector_name = "IBEA"


class NSGA2_BLUEPYOPT(BluepyoptAlgorithmBasis):
	def __init__(self, reader_obj,  option_obj):
		BluepyoptAlgorithmBasis.__init__(self, reader_obj,  option_obj)
		self.selector_name = "NSGA2"