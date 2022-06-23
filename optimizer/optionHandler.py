
import time
from xml.etree.ElementTree import Element as e, SubElement as se
from xml.etree import ElementTree
from xml.dom import minidom
import pickle as pickle
import re
import collections

class OrderedSet(collections.MutableSet):

    def __init__(self, iterable=None):
        self.end = end = [] 
        end += [None, end, end]         # sentinel node for doubly linked list
        self.map = {}                   # key --> [key, prev, next]
        if iterable is not None:
            self |= iterable

    def __len__(self):
        return len(self.map)

    def __contains__(self, key):
        return key in self.map

    def add(self, key):
        if key not in self.map:
            end = self.end
            curr = end[1]
            curr[2] = end[1] = self.map[key] = [key, curr, end]

    def discard(self, key):
        if key in self.map:        
            key, prev, next = self.map.pop(key)
            prev[2] = next
            next[1] = prev

    def __iter__(self):
        end = self.end
        curr = end[2]
        while curr is not end:
            yield curr[0]
            curr = curr[2]

    def __reversed__(self):
        end = self.end
        curr = end[1]
        while curr is not end:
            yield curr[0]
            curr = curr[1]

    def pop(self, last=True):
        if not self:
            raise KeyError('set is empty')
        key = self.end[1][0] if last else self.end[2][0]
        self.discard(key)
        return key

    def __repr__(self):
        if not self:
            return '%s()' % (self.__class__.__name__,)
        return '%s(%r)' % (self.__class__.__name__, list(self))

    def __eq__(self, other):
        if isinstance(other, OrderedSet):
            return len(self) == len(other) and list(self) == list(other)
        return set(self) == set(other)

def meanstdv(x):
	from math import sqrt
	n, mean, std = len(x), 0, 0
	for a in x:
		mean = mean + a
	mean = mean / float(n)
	for a in x:
		std = std + (a - mean)**2
	std = sqrt(std / float(n-1))
	return mean, std



# class to handle the settings specified by the user
# there are no separate classes for the different settings, only get-set member functions
# the proper initialization is done via the target classes' constructors (traceReader, modelHandlerNeuron)
class optionHandler(object):
	"""
	Object to store the settings required by the optimization work flow.
	"""
	def __init__(self):
		self.output_level="0"
		prev=dir(self)
		self.start_time_stamp=time.time()
		#exp data settings
		self.base_dir="" # path to base directory
		self.input_dir="" # path to input file
		self.input_size=0 # no_traces
		self.input_scale="" # scale of input
		self.input_length=1 # length of input
		self.input_freq=1 # sampling freq of input
		self.input_cont_t=0 # contains time or not
		self.type=[]

		#model file settings
		self.model_path="" # path to the model file (.hoc)
		self.model_spec_dir="" #path to the channel files

		self.u_fun_string=""#string of the user defined function waiting for compilation

		self.simulator=""
		self.sim_command=""

		#stim settings
		self.stim_type="" # type of stimulus
		self.stim_pos=0 # position
		self.stim_sec="" # section name
		self.stim_amp=[] # stimuli amplitude
		self.stim_del=0 # delay
		self.stim_dur=0 # duration

		#parameters and values
		self.adjusted_params=[] # string list of the editable things, section, channel, parameter
		self.param_vals=[] # list of values to the parameters

		#run controll settings
		self.run_controll_tstop=0 # tstop
		self.run_controll_dt=0 # dt
		self.run_controll_record="" # parameter to be recorded
		self.run_controll_sec="" # section where the recording takes place
		self.run_controll_pos=0 # position where the recording takes place
		self.run_controll_vrest=0 # resting voltage

		#optimizer settings
		self.seed=None
		self.current_algorithm=None

		self.pop_size=None
		self.number_of_cpu=None
		self.num_islands=None
		self.max_evaluation=None
		self.mutation_rate=None
		self.crossover_rate=None
		self.force_bounds=None

		self.cooling_rate=None
		self.m_gauss=None
		self.std_gauss=None
		self.step_size=None
		self.init_temp=None
		self.temperature=None
		self.acc=None
		self.update_freq=None
		self.num_iter=None
		self.num_repet=None

		self.x_tol=None
		self.f_tol=None

		self.inertia=None
		self.cognitive_rate=None
		self.social_rate=None
		self.neighborhood_size=None
		#self.topology=None

		self.num_params=None
		self.boundaries=[[],[]]
		self.starting_points=None

		self.spike_thres=0
		self.spike_window=50
		#self.ffunction=None #other parameters might be necessary
		self.feats=[]
		self.weights=[]
		post=dir(self)
		self.class_content=list(OrderedSet(post)-OrderedSet(prev))

		self.algorithm_parameters_dict={"GA_INSPYRED":{"num_selected":self.pop_size,"crossover_rate":1,"num_crossover_points":1,"mutation_rate":1,"num_elites":0},
		"CES_INSPYRED":{"tau":None,"tau_prime":None,"epsilon":0.00001},
		"EDA_INSPYRED":{"num_selected":2,"num_offspring":self.pop_size,"num_elites":0},
		"DEA_INSPYRED":{"num_selected":2,"tournament_size":2,"crossover_rate":1,"mutation_rate":0.1,"gaussian_mean":0,"gaussian_stdev":1},
		"SA_INSPYRED":{"temperature":None,"cooling_rate":None,"mutation_rate":None,"gaussian_mean":0,"gaussian_stdev":1},
		"ACO_INSPYRED":{"initial_pheromone":0,"evaporation_rate":0.1,"learning_rate":0.1},
		"PSO_INSPYRED":{"inertia":0.5,"cognitive_rate":2.1,"social_rate":2.1},
		"GACO_PYGMO":{"ker":63,"q":1.0,"oracle":0.,"acc":0.01,"threshold":1,"n_gen_mark":7,"impstop":100000,"evalstop":100000,"focus":0.,"memory":False},
		"DE_PYGMO":{"F":0.8,"CR":0.9,"variant":2,"f_tol":1e-6,"x_tol":1e-6},
		"SADE_PYGMO":{"variant":2, "variant_adptv":1, "f_tol":1e-06, "x_tol":1e-06, "memory":False},
		"DE1220_PYGMO":{"allowed_variants":[2, 3, 7, 10, 13, 14, 15, 16], "variant_adptv":1, "f_tol":1e-06, "x_tol":1e-06, "memory":False},
		"PSO_PYGMO":{"omega":0.7298, "eta1":2.05, "eta2":2.05, "max_vel":0.5, "variant":5, "neighb_type":2, "neighb_param":4, "memory":False},
		"PSOG_PYGMO":{"omega":0.7298, "eta1":2.05, "eta2":2.05, "max_vel":0.5, "variant":5, "neighb_type":2, "neighb_param":4, "memory":False},
		"SGA_PYGMO":{"cr":0.9, "eta_c":1.0, "m":0.02, "param_m":1.0, "param_s":2},
		"ABC_PYGMO":{"limit":1},
		"CMAES_PYGMO":{"cc":- 1, "cs":- 1, "c1":- 1, "cmu":- 1, "sigma0":0.5, "f_tol":1e-06, "x_tol":1e-06, "memory":False, "force_bounds":False},
		"XNES_PYGMO":{"eta_mu":- 1, "eta_sigma":- 1, "eta_b":- 1, "sigma0":- 1, "f_tol":1e-06, "x_tol":1e-06, "memory":False, "force_bounds":False},
		"NSGA_PYGMO":{"cr":0.95, "eta_c":10.0, "m":0.01, "eta_m":50.0},
		"MACO_PYGMO":{"ker":63, "q":1.0, "threshold":1, "n_gen_mark":7, "evalstop":100000, "focus":0.0, "memory":False},
		"NSPSO_PYGMO":{"omega":0.6, "c1":0.01, "c2":0.5, "chi":0.5, "v_coeff":0.5, "leader_selection_range":2, "memory":False},
		"CMAES_CMAES":{"sigma":1.3},
		"PRAXIS_PYGMO":{},
		"NM_PYGMO":{},
		"BH_SCIPY":{"T":1.0, "stepsize":0.5, "interval":50, "disp":False,"target_accept_rate":0.5, "stepwise_factor":0.9},
		"NM_SCIPY":{'disp': False, 'xatol': 0.0001, 'fatol': 0.0001, 'adaptive': False},
		"L_BFGS_B_SCIPY":{'maxcor': 10, 'ftol': 2.220446049250313e-09, 'gtol': 1e-05, 'eps': 1e-08},
		"RANDOM_SEARCH":{},
		"NSGA2_BLUEPYOPT":{'mutpb':1.0, 'cxpb':1.0},
		"IBEA_BLUEPYOPT":{'mutpb':1.0, 'cxpb':1.0},
		}

	"""def dump(self,f_mapper):
		
		Dumps the content of the class into a string.

		:param f_mapper: a ``dictionary`` that maps the fitness function objects to their names (used in the GUI)

		:return: the content of the class as ``string``

		

		root=e("settings")
		for m in self.class_content:
			child=se(root,m)
			try:
				if m=="feats":
					if self.type[-1]!='features':
						child.text=", ".join([f_mapper[x.__name__] for x in self.__getattribute__(m)])
					else:
						child.text=", ".join(self.feats)
				else:
					child.text=str(self.__getattribute__(m).__str__())
					if child.text=='':
						child.text="\"\""
			except TypeError:
				child.text="None"
		
		return prettify(root)"""


	def create_dict_for_json(self,f_mapper):
		json_dict={}
		for m in self.class_content:
			if m=="feats":
				if self.type[-1]!='features':
					json_dict[m]=[f_mapper[x.__name__] for x in self.__getattribute__(m)]
				else:
					json_dict[m]=self.feats
			else:
				json_dict[m]=self.__getattribute__(m)
		
		return {"selectable_algorithms":self.algorithm_parameters_dict,"attributes":json_dict}


	def read_all_json(self,settings):
		for key, value in settings.items():
			self.__setattr__(key,value)
		if isinstance(self.current_algorithm, str):
			self.algorithm_name=re.sub('_+',"_",re.sub("[\(\[].*?[\)\]]", "", self.current_algorithm).replace("-","_").replace(" ","_")).upper()
			self.algorithm_parameters={}
		else:
			self.algorithm_name=re.sub('_+',"_",re.sub("[\(\[].*?[\)\]]", "", list(self.current_algorithm.keys())[0]).replace("-","_").replace(" ","_")).upper()
			self.algorithm_parameters=list(self.current_algorithm.values())[0]
		


	# returns the current settings of the current working directory (referred as base in modelHandler, used in traceReader )
	def GetFileOption(self):
		"""
		:return: the current working directory (referred as base in modelHandler, used in traceReader )

		"""
		return self.base_dir

	# sets the current working directory, and other directory specific settings to the given value(s)
	def SetFileOptions(self,options):
		"""
		Sets the current working directory

		:param options: the path of the directory

		"""
		self.base_dir=options

	# returns the current input file options
	def GetInputOptions(self):
		"""
		Gets the input related settings:
			* input file
			* number of traces in file
			* unit of input
			* length of the individual traces (see traceHandler)
			* sampling frequency of the trace(s)
			* flag indicating if file included time scale or not (will be removed, see traceHandler)
			* the type of the trace(s)

		:return: the parameters listed above in a ``list``

		"""
		return [self.input_dir,
				self.input_size,
				self.input_scale,
				self.input_length,
				self.input_freq,
				self.input_cont_t,
				self.type[-1]]

	# sets the input file options to the given values
	def SetInputOptions(self,options):
		"""
		Sets the options related to the input to the given values.

		:param options: a ``list`` of values (order of parameter should be the same as listed in ``GetInputOptions``)

		"""
		self.input_dir=options[0]
		self.input_size=options[1]
		self.input_scale=options[2]
		self.input_length=options[3]
		self.input_freq=options[4]
		self.input_cont_t=options[5]
		self.type.append(options[6])

	def GetSimParam(self):
		"""
		Gets the simulator related parameters:
			* the name of the simulator
			* the command which should be executed to run the model (see modelHandler)

		:return: the parameters listed above in a ``list``

		"""
		return [self.simulator,self.sim_command]

	def SetSimParam(self,options):
		"""
		Sets the simulator related parameters.

		:param options: a ``list`` of values

		"""
		self.simulator=options[0]
		self.sim_command=options[1]


	def GetModelOptions(self):
		"""
		Gets the model related options:
			* path to the model
			* path to the directory containing the special files (see modelHanlder)

		:return: the parameters listed above in a ``list``

		"""
		return [self.model_path,
		self.model_spec_dir]

	def SetModelOptions(self,options):
		"""
		Sets the model related options.

		:param options: a ``list`` of values

		"""
		self.model_path=options[0]
		self.model_spec_dir=options[1]

	def GetUFunString(self):
		"""
		Gets the user defined function.

		:return: the function as a ``string``

		"""
		return self.u_fun_string.strip("\"")

	def SetUFunString(self,s):
		"""
		Sets the user defined function.

		:param s: the function as a ``string``

		"""
		self.u_fun_string=s

	def GetModelStim(self):
		"""
		Gets the parameters regarding the stimulus type:
			* type of the stimulus
			* position of stimulus
			* name of the stimulated section

		:return: the parameters listed above in a ``list``

		"""
		return [self.stim_type,
		self.stim_pos,
		self.stim_sec]

	def SetModelStim(self,options):
		"""
		Sets the parameters regarding the stimulus type to the given values.

		:param options: ``list`` of values

		"""
		self.stim_type=options[0]
		self.stim_pos=options[1]
		self.stim_sec=options[2]

	def GetModelStimParam(self):
		"""
		Gets the parameters of the stimulus:
			* amplitude
			* delay
			* duration

		:return: the parameters listed above in a ``list``

		"""
		
		#pickle.dump([self.stim_amp,self.stim_del,self.stim_dur], open(self.GetFileOption() + "/stim.p", "wb" ) )
		
		return [self.stim_amp,
		self.stim_del,
		self.stim_dur]

	def SetModelStimParam(self,options):
		"""
		Sets the parameters of the stimulus to the given values.

		:param options: ``list`` of values

		.. note::
			Only the parameters of the IClamp are stored this way since the parameters of the
			SEClamp are obtained by combining the values here and the values regarding the simulation.

		"""
		self.stim_amp=options[0]
		self.stim_del=options[1]
		self.stim_dur=options[2]

	def GetObjTOOpt(self):
		"""
		Gets the parameters selected to optimization.

		:return: a ``list`` of ``strings``

		"""
		return self.adjusted_params

	def SetObjTOOpt(self,options):
		"""
		Adds the given parameter to the list of parameters selected for optimization.

		:param options: a ``string`` containing the section, a channel name and a channel parameter name,
			or a morphological parameter separated by spaces

		.. note::
			If a given parameter is already stored then it will not added to the list.

		"""
		if self.adjusted_params.count(options)==0:
			self.adjusted_params.append(options)#string list, one row contains the section, the channel, and the parameter name
			#print options
		else:
			print("already selected\n")
		#self.adjusted_params=list(set(self.adjusted_params))

	def GetOptParam(self):
		"""
		Not in use!
		Gets the list of parameter values corresponding to the parameters subject to optimization.

		:return: ``list`` of real values

		"""
		return self.param_vals

	def SetOptParam(self,options):
		"""
		Not in use!
		Adds the given value to the list of parameter values corresponding to the parameters subject to optimization.

		:param options: a real value

		"""
		self.param_vals.append(options)#float list, with all the values which selected for optimization

	def GetModelRun(self):
		"""
		Gets the parameters corresponding to the simulation:
			* length of simulation
			* integration step
			* parameter to record
			* section name
			* position inside the section
			* initial voltage

		:return: the parameters above in a ``list``

		"""
		#pickle.dump([self.run_controll_tstop,self.run_controll_dt,self.run_controll_record,self.run_controll_sec,self.run_controll_pos,self.run_controll_vrest], open( "estim.p", "wb" ) )
		return [self.run_controll_tstop,
		self.run_controll_dt,
		self.run_controll_record,
		self.run_controll_sec,
		self.run_controll_pos,
		self.run_controll_vrest]

	def SetModelRun(self,options):
		"""
		Sets the parameters regarding the simulation to the given values.

		:param options:  ``list`` of parameters

		"""
		self.run_controll_tstop=options[0]
		self.run_controll_dt=options[1]
		self.run_controll_record=options[2]
		self.run_controll_sec=options[3]
		self.run_controll_pos=options[4]
		self.run_controll_vrest=options[5]

	def GetOptimizerOptions(self):
		"""
		Gets the parameters regarding the optimization process:
			* seed: random seed
			* current_algorithm: name of evolution algorithm
			* Size of Population: size of population
			* Number of Generations: number of generations
			* Mutation Rate: mutation rate (0-1)
			* Cooling Rate: cooling rate (0-1)
			* Mean of Gaussian: mean value of gaussian
			* Std. Deviation of Gaussian: standard deviation of gaussian
			* Cooling Schedule: index of cooling schedule
			* Initial Temperature: initial temperature
			* Final Temperature: final temperature
			* Accuracy: accuracy
			* Dwell: number of evaluation on the given temperature level
			* Error Tolerance for x: error tolerance for input values
			* Error Tolerance for f: error tolerance for fitness values
			* num_params: number of input parameters
			* boundaries: bounds of the parameters
			* starting_points: initial values to the algorithm

		:return: a ``dictionary`` containing the parameters above

		"""
		return {"seed" : self.seed,
				"current_algorithm" : self.current_algorithm,
				"Size of Population" : self.pop_size,
				"Number of Islands" : self.num_islands,
				"Number of Generations" : self.max_evaluation,
				"Force bounds" : self.force_bounds,
				"Mutation Rate" : self.mutation_rate,
				"Crossover Rate" : self.crossover_rate,
				"Cooling Rate" : self.cooling_rate,
				"Mean of Gaussian" : self.m_gauss,
				"Std. Deviation of Gaussian" : self.std_gauss,
				"Initial Temperature" : self.init_temp,
				"Step Size" : self.step_size,
				"Temperature" : self.temperature,
				"Update Frequency" : self.update_freq,
				"Number of Iterations" : self.num_iter,
				"Number of Repetition" : self.num_repet,
				"Error Tolerance for x" : self.x_tol,
				"Error Tolerance for f" : self.f_tol,
				"num_params" : self.num_params,
				"boundaries" : self.boundaries,
				"starting_points" : self.starting_points,
				"Inertia" :self.inertia,
				"Cognitive Rate" : self.cognitive_rate,
				"Social Rate" : self.social_rate,
				"Neighborhood Size" : self.neighborhood_size,
				"Number of CPU" : self.number_of_cpu
		#"Topology:" : self.topology
				}

	# sets the optimizer settings (which optimizer, fitness function, generator settings, etc)
	def SetOptimizerOptions(self,options):
		"""
		Sets the parameters regarding the optimization process.

		:param options: a ``dictionary`` containing the parameters

		"""
		self.seed=options.get("seed",1234)
		self.current_algorithm=options.get("current_algorithm")
		self.algorithm_name=re.sub('_+',"_",re.sub("[\(\[].*?[\)\]]", "", self.current_algorithm).replace("-","_").replace(" ","_")).upper()
		self.algorithm_parameters={}
		self.pop_size=options.get("Size of Population",None)
		self.num_islands=options.get("Number of Islands", None)
		self.max_evaluation=options.get("Number of Generations",None)
		self.mutation_rate=options.get("Mutation Rate",None)
		self.crossover_rate=options.get("Crossover Rate:",None)
		self.force_bounds=options.get("Force bounds",False)
		self.cooling_rate=options.get("Cooling Rate",None)

		self.m_gauss=options.get("Mean of Gaussian",None)
		self.std_gauss=options.get("Std. Deviation of Gaussian",None)
		self.init_temp=options.get("Initial Temperature",None)
		self.step_size=options.get("Step Size",None)
		self.temperature=options.get("Temperature",None)

		self.acc=options.get("Accuracy",None)
		self.update_freq=options.get("Update Frequency",None)
		self.num_iter=options.get('Number of Iterations',None)
		self.num_repet=options.get('Number of Repetition',None)

		self.x_tol=options.get("Error Tolerance for x",None)
		self.f_tol=options.get("Error Tolerance for f",None)

		self.num_params=options.get("num_params")
		self.boundaries=options.get("boundaries")
		self.starting_points=options.get("starting_points",None)

		self.inertia=options.get("Inertia",None)
		self.cognitive_rate=options.get("Cognitive Rate",None)
		self.social_rate=options.get("Social Rate",None)
		self.neighborhood_size=options.get("Neighborhood Size",None)

		self.number_of_cpu=options.get("Number of CPU",None)
	#self.topology=options.GetCurrentSelection("Topology:")


	def GetFitnessParam(self):
		"""
		Gets the parameters required by the fitness functions:
			* ``list`` consisting of:
				* a ``dictionary`` containing the spike detection threshold and the spike window
				* a ``list`` of fitness function names
			* ``list`` of weights to combine the fitness functions

		:return: a ``list`` containing the structures described above

		"""
		return [ [{"Spike Detection Thres. (mv)" : self.spike_thres,
				"Spike Window (ms)" : self.spike_window},self.feats],self.weights ]

	def SetFitnesParam(self,options):
		"""
		Sets the parameters required by the fitness functions.

		:param options: the required values in the structure described in ``GetFitnessParam``

		"""
		self.spike_thres=options[0][0].get("Spike Detection Thres. (mv)",0.0)
		# in the case of abstract data (features) input_freq is not given
		if self.input_freq!= None:
			self.spike_window=options[0][0].get("Spike Window (ms)",1)*self.input_freq/1000.0
		else:
			self.spike_window=None
		#self.ffunction=options[0][1]
		self.feats=options[0][1]
		self.weights=options[1]
