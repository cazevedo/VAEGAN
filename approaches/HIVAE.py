import tensorflow as tf
from .libHIVAE import parser_arguments, read_functions, graph_new
import time
import numpy as np
import os


def test(dataset):
	settings = '--epochs 100 --model_name model_HIVAE_inputDropout --restore 0 --train 1 \
	            --types_file defaultCredit/data_types.csv \
	            --batch_size 1000 --save 1001 --save_file model_test\
	            --dim_latent_s 10 --dim_latent_z 10 --dim_latent_y 5 \
	            --miss_percentage_train 0.2 --miss_percentage_test 0.5'

	argvals = settings.split()
	args = parser_arguments.getArgs(argvals)

	#Create a directoy for the save file
	if not os.path.exists('./Saved_Networks/' + args.save_file):
	    os.makedirs('./Saved_Networks/' + args.save_file)

	network_file_name='./Saved_Networks/' + args.save_file + '/' + args.save_file +'.ckpt'
	log_file_name='./Saved_Network/' + args.save_file + '/log_file_' + args.save_file +'.txt'

	print(args)

	# Transform given dataset to the format we need
	original_dataset, train_pd, types_dict, miss_mask, true_miss_mask, n_samples = read_functions.from_custom(dataset, 0)
	train_data = train_pd.values

	#Creating graph
	sess_HVAE = tf.Graph()

	with sess_HVAE.as_default():
	    tf_nodes = graph_new.HVAE_graph(args.model_name, types_dict, args.batch_size,
	                                learning_rate=1e-3, z_dim=args.dim_latent_z, y_dim=args.dim_latent_y, s_dim=args.dim_latent_s, y_dim_partition=args.dim_latent_y_partition)

	return tf_nodes, train_pd, train_data, types_dict, miss_mask, true_miss_mask, n_samples
