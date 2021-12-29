{"GA_Inspyred":{"num_selected":len(population),"crossover_rate":1,"num_crossover_points":1,"mutation_rate":1,"num_elites":0},
"CES_Inspyred":{"tau":None,"tau_prime":None,"epsilon":0.00001},
"EDA_Inspyred":{"num_selected":len(population)/2,"num_offspring":len(population),"num_elites":0},
"DEA_Inspyred":{"num_selected":2,"tournament_size":2,"crossover_rate":1,"mutation_rate":0.1,"gaussian_mean":0,"gaussian_stdev":1},
"SA_Inspyred":{"temperature":None,"cooling_rate":None,"mutation_rate":None,"gaussian_mean":0,"gaussian_stdev":1},
"ACO_Inspyred":{"initial_pheromone":0,"evaporation_rate":0.1,"learning_rate":0.1},
"PSO_Inspyred":{"inertia":0.5,"cognitive_rate":2.1,"social_rate":2.1},
"GACO_Pygmo":{"ker":63,"q":1.0,"oracle":0.,"acc":0.01,"threshold":1,"n_gen_mark":7,"impstop":100000,"evalstop":100000,"focus":0.,"memory":False},
"DE_Pygmo":{"F":0.8,"CR":0.9,"variant":2,"f_tol":1e-6,"x_tol":1e-6},
"SADE_Pygmo":{"variant":2, "variant_adptv":1, "f_tol":1e-06, "x_tol":1e-06, "memory":False},
"DE1220_Pygmo":{"allowed_variants":[2, 3, 7, 10, 13, 14, 15, 16], "variant_adptv":1, "f_tol":1e-06, "x_tol":1e-06, "memory":False},
"PSO_Pygmo":{"inertia":0.7298, "social_rate":2.05, "cognitive_rate":2.05, "max_vel":0.5, "variant":5, "neighb_type":2, "neighb_param":4, "memory":False},
"PSOG_Pygmo":{"inertia":0.7298, "social_rate":2.05, "cognitive_rate":2.05, "max_vel":0.5, "variant":5, "neighb_type":2, "neighb_param":4, "memory":False},
"SGA_Pygmo":{"cr":0.9, "eta_c":1.0, "m":0.02, "param_m":1.0, "param_s":2, "crossover":'exponential', "mutation":'polynomial', "selection":'tournament'},
"ABC_Pygmo":{"limit":1},
"CMAES_Pygmo":{"cc":- 1, "cs":- 1, "c1":- 1, "cmu":- 1, "sigma0":0.5, "f_tol":1e-06, "x_tol":1e-06, "memory":False, "force_bounds":False},
"XNES_Pygmo":{"eta_mu":- 1, "eta_sigma":- 1, "eta_b":- 1, "sigma0":- 1, "f_tol":1e-06, "x_tol":1e-06, "memory":False, "force_bounds":False},
"NSGA_Pygmo":{"cr":0.95, "eta_c":10.0, "m":0.01, "eta_m":50.0},
"MACO_Pygmo":{"ker":63, "q":1.0, "threshold":1, "n_gen_mark":7, "evalstop":100000, "focus":0.0, "memory":False},
"NSPSO_Pygmo":{"inertia":0.6, "c1":0.01, "c2":0.5, "chi":0.5, "v_coeff":0.5, "leader_selection_range":2, "diversity_mechanism":'crowding distance', "memory":false}
}
