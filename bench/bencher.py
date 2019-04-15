import subprocess
import datetime
import os
import json 
import sys 
import getopt

def analyze_singleton_experiment(input_file):
	with open(input_file) as json_file:
		data = json.load(json_file)
		print("GPU hardware: %s" % (data["slab_hash"]['device_name']))
		trials = data["slab_hash"]["trial"]

		for trial in trials:
			data_q0 = (trial["load_factor"], trial["build_rate_mps"], trial["search_rate_mps"], trial["search_rate_bulk_mps"])

		print("===============================================================================================")
		print("Singleton experiment:")
		print("\tNumber of elements to be inserted: %d" % (trials[0]['num_keys']))
		print("\tNumber of buckets: %d" % (trials[0]['num_buckets']))
		print("\tExpected chain length: %.2f" % (trials[0]['exp_chain_length']))
		print("===============================================================================================")
		print("load factor\tbuild rate(M/s)\t\tsearch rate(M/s)\tsearch rate bulk(M/s)")
		print("===============================================================================================")
		print("%.2f\t\t%.3f\t\t%.3f\t\t%.3f" % (data_q0[0], data_q0[1], data_q0[2], data_q0[3]))

def analyze_load_factor_experiment(input_file):
	with open(input_file) as json_file:
		data = json.load(json_file)
		print("GPU hardware: %s" % (data["slab_hash"]['device_name']))
		trials = data["slab_hash"]["trial"]

		tabular_data = []

		for trial in trials:
			tabular_data.append((trial["load_factor"], 
				trial["build_rate_mps"], 
				trial["search_rate_mps"], 
				trial["search_rate_bulk_mps"], 
				trial['num_buckets']))

		tabular_data.sort()
		print("===============================================================================================")
		print("Load factor experiment:")
		print("\tTotal number of elements is fixed, load factor (number of buckets) is a variable")
		print("\tNumber of elements to be inserted: %d" % (trials[0]['num_keys']))
		print("\t %.2f of %d queries exist in the data structure" % (trials[0]['query_ratio'], trials[0]['num_queries']))
		print("===============================================================================================")
		print("load factor\tnum buckets\tbuild rate(M/s)\t\tsearch rate(M/s)\tsearch rate bulk(M/s)")
		print("===============================================================================================")
		for pair in tabular_data:
			print("%.2f\t\t%d\t\t%.3f\t\t%.3f\t\t%.3f" % (pair[0], pair[4], pair[1], pair[2], pair[3]))		

def analyze_table_size_experiment(input_file):
	with open(input_file) as json_file:
		data = json.load(json_file)
		print("GPU hardware: %s" % (data["slab_hash"]['device_name']))
		trials = data["slab_hash"]["trial"]

		tabular_data = []

		for trial in trials:
			tabular_data.append((trial["num_keys"], 
				trial['num_buckets'],
				trial['load_factor'], 
				trial["build_rate_mps"], 
				trial["search_rate_mps"], 
				trial["search_rate_bulk_mps"]))

		tabular_data.sort()
		print("===============================================================================================")
		print("Table size experiment:")
		print("\tTable's expected chain length is fixed, and total number of elements is variable")
		print("\tExpected chain length = %.2f\n" % trials[0]['exp_chain_length'])
		print("\t%.2f of %d queries exist in the data structure" % (trials[0]['query_ratio'], trials[0]['num_queries']))
		print("===============================================================================================")
		print("(num keys, num buckets, load factor)\tbuild rate(M/s)\t\tsearch rate(M/s)\tsearch rate bulk(M/s)")
		print("===============================================================================================")
		for pair in tabular_data:
			print("(%d, %d, %.2f)\t\t\t%10.3f\t\t%.3f\t\t%.3f" % (pair[0], pair[1], pair[2], pair[3], pair[4], pair[5]))

def analyze_concurrent_experiment(input_file):
	with open(input_file) as json_file:
		data = json.load(json_file)
		print("GPU hardware: %s" % (data["slab_hash"]['device_name']))
		trials = data["slab_hash"]["trial"]

		tabular_data = []

		for trial in trials:
			tabular_data.append((trial["init_load_factor"], 
				trial['final_load_factor'], 
				trial['num_buckets'], 
				trial["initial_rate_mps"], 
				trial["concurrent_rate_mps"]))

		tabular_data.sort()
		print("===============================================================================================")
		print("Concurrent experiment:")
		print("\tvariable load factor, fixed number of elements")
		print("\tOperation ratio: (insert, delete, search) = (%.2f, %.2f, [%.2f, %.2f])" % (trials[0]['insert_ratio'], trials[0]['delete_ratio'], trials[0]['search_exist_ratio'], trials[0]['search_non_exist_ratio']))
		print("===============================================================================================")
		print("batch_size = %d, init num batches = %d, final num batches = %d" % (trials[0]['batch_size'], trials[0]['num_init_batches'], trials[0]['num_batches']))
		print("===============================================================================================")
		print("init lf\t\tfinal lf\tnum buckets\tinit build rate(M/s)\tconcurrent rate(Mop/s)")
		print("===============================================================================================")
		for pair in tabular_data:
			print("%.2f\t\t%.2f\t\t%d\t\t%.3f\t\t%.3f" % (pair[0], pair[1], pair[2], pair[3], pair[4]))					

def main(argv):
	input_file = ''
	try:
		opts, args = getopt.getopt(argv, "hvi:m:d:", ["help", "verbose", "ifile=", "mode=", "device="])
	except getopt.GetOptError:
		print("bencher.py -i <inputfile> -m <experiment mode> -d <device index> -v")
		sys.exit(2)
	
	for opt, arg in opts:
		if opt == '-h':
			print("===============================================================================================")
			print("-i/--ifile: 	\t\t Input file (optional)")
			print("-m/--mode: 	\t\t Experiment mode:")
			print("\t\t\t\t\t 0: singleton experiment")
			print("\t\t\t\t\t 1: load factor experiment")
			print("\t\t\t\t\t 2: variable sized table experiment")
			print("\t\t\t\t\t 3: concurrent experiment")
			print("-v/--verbose")
			print("===============================================================================================")
			sys.exit()
		else:
			if opt in ("-i", "--ifile"):
				input_file = arg
				print("input file: " + input_file)
			if opt in ("-m", "--mode"):
				mode = int(arg)
			if opt in ("-d", "--device"):
				device_idx = int(arg)
			if opt in ("-v", "--verbose"):
				verbose = True
			else:
				verbose =  False
	
	# if the input file is not given, proper experiments should be run first
	if not input_file:		
		# == creating a folder to store results
		out_directory = "../build/bench_result/"
		if (not os.path.isdir(out_directory)):
			os.mkdir(out_directory)

		# == running benchmark files
		bin_file = "../build/bin/benchmark"
		if(not os.path.exists(bin_file)):
			raise Exception("binary file " + bin_file + " not found!")

		# creating a unique name for the file
		cur_time_list = str(datetime.datetime.now()).split()
		out_file_name = "out"
		for s in cur_time_list:
			out_file_name += ("_" + s)

		out_file_dest = out_directory + out_file_name + ".json"
		input_file = out_file_dest # input file for the next step
		print("intermediate results stored at: " + out_file_dest)

		print("mode = %d" % mode)
		if mode == 0:
			args = (bin_file, "-mode", str(mode), 
				"-num_key", str(2**22),
				"-expected_chain", str(0.6),
				"-device", str(device_idx),
				"-filename", out_file_dest,
				"-verbose", "1" if verbose else "0")
		elif mode == 1:
			args = (bin_file, 
				"-mode", str(mode),
				"-num_keys", str(2**22),
				"-quary_ratio", str(1.0),
				"-device", str(device_idx),
				"-lf_bulk_step", str(0.1),
				"-lf_bulk_num_sample", str(20), 
				"-filename", out_file_dest,
				"-verbose", "1" if verbose else "0")
		elif mode == 2:
				args = (bin_file, "-mode", str(mode), 
				"-nStart", str(18), 
				"-nEnd", str(23), 
				"-expected_chain", str(0.6),
				"-query_ratio", str(1.0),
				"-device", str(device_idx),
				"-filename", out_file_dest,
				"-verbose", "1" if verbose else "0")
		elif mode == 3:
			args = (bin_file, "-mode", str(mode),
			"-nStart", str(18),
			"-nEnd", str(21),
			"-num_batch", str(4),
			"-init_batch", str(3),
			"-lf_conc_step", str(0.1),
			"-lf_conc_num_sample", str(10),
			"-device", str(device_idx), 
			"-filename", out_file_dest,
			"-verbose", "1" if verbose else "0")

		print(" === Started benchmarking ... ")

		popen = subprocess.Popen(args, stdout = subprocess.PIPE)
		popen.wait()

		if verbose:
			output = popen.stdout.read()
			print(output)
		print(" === Done!")
	elif not os.path.exists(input_file):
		raise Exception("Input file " + input_file + " does not exist!")

	# reading the json files:
	if mode == 0:
		analyze_singleton_experiment(input_file)
	elif mode == 1:
		analyze_load_factor_experiment(input_file)
	elif mode == 2:
		analyze_table_size_experiment(input_file)
	elif mode == 3:
		analyze_concurrent_experiment(input_file)	
	else:
		print("Invalid mode entered")
		sys.exit(2)

if __name__ == "__main__":
	main(sys.argv[1:])					