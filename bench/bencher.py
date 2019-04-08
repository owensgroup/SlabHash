import subprocess
import datetime
import os
import json 
import sys 
import getopt
# import matplotlib.pyplot as plt 

def main(argv):
	input_file = ''
	try:
		opts, args = getopt.getopt(argv, "hi:", ["help", "ifile="])
	except getopt.GetOptError:
		print("bencher.py -i <inputfile>")
		sys.exit(2)
	
	for opt, arg in opts:
		if opt == '-h':
			print("bencher.py -i <inputfile>")
			sys.exit()
		elif opt in ("-i", "--ifile"):
			input_file = arg

			print(" === " + input_file)
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
		print(" == filename = " + out_file_dest)

		args = (bin_file, "-mode", "1", "-filename", out_file_dest)

		print(" === Started benchmarking ... ")

		popen = subprocess.Popen(args, stdout = subprocess.PIPE)
		popen.wait()

		output = popen.stdout.read()
		print(output)
		print(" === Done!")
	elif not os.path.exists(input_file):
		raise Exception("Input file " + input_file + " does not exist!")

	# reading the json files:
	with open(input_file) as json_file:
		data = json.load(json_file)
		print(data["slab_hash"]['device_name'])
		trials = data["slab_hash"]["trial"]

		tabular_data_q0 = []
		tabular_data_q1 = []

		for trial in trials:
			if( abs(trial["query_ratio"]) < 0.000001):
				tabular_data_q0.append((trial["load_factor"], trial["build_rate_mps"], trial["search_rate_mps"], trial["search_rate_bulk_mps"]))
			elif abs(trial["query_ratio"] - 1.0) < 0.000001:
				tabular_data_q1.append((trial["load_factor"], trial["build_rate_mps"], trial["search_rate_mps"], trial["search_rate_bulk_mps"]))

		tabular_data_q0.sort()
		print("Experiments when none of the queries exist:")
		print("load factor\tbuild rate(M/s)\t\tsearch rate(M/s)\tsearch rate bulk(M/s)")
		for pair in tabular_data_q0:
			print("%.2f\t\t%.3f\t\t%.3f\t\t%.3f" % (pair[0], pair[1], pair[2], pair[3]))		

		tabular_data_q1.sort()
		print("Experiments when all of the queries exist:")
		print("load factor\tbuild rate(M/s)\t\tsearch rate(M/s)\tsearch rate bulk(M/s)")
		for pair in tabular_data_q1:
			print("%.2f\t\t%.3f\t\t%.3f\t\t%.3f" % (pair[0], pair[1], pair[2], pair[3]))		

if __name__ == "__main__":
	main(sys.argv[1:])