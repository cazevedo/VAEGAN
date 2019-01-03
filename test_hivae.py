import os
from get_datasets import credit_example
from approaches import HIVAE
from approaches.libHIVAE import parser_arguments, read_functions
from approaches.libHIVAE import main as old_hivae

if __name__ == "__main__":

    # old_hivae.run()

    settings = '--epochs 100 --model_name model_HIVAE_inputDropout --restore 0 --train 1 \
                --types_file defaultCredit/data_types.csv \
                --batch_size 1000 --save 1001 --save_file model_test\
                --dim_latent_s 10 --dim_latent_z 10 --dim_latent_y 5 \
                --miss_percentage_train 0.2 --miss_percentage_test 0.5'

    argvals = settings.split()
    args = parser_arguments.getArgs(argvals)
    print(args)

    #Create a directoy for the save file
    if not os.path.exists('./Saved_Networks/' + args.save_file):
        os.makedirs('./Saved_Networks/' + args.save_file)

    network_file_name='./Saved_Networks/' + args.save_file + '/' + args.save_file +'.ckpt'
    log_file_name='./Saved_Network/' + args.save_file + '/log_file_' + args.save_file +'.txt'

    credit = credit_example()
    config_idx = 0

    # Transform given dataset to the format we need
    original_dataset, train_pd, types_dict, miss_mask, true_miss_mask, n_samples = read_functions.from_custom(credit, config_idx)

    # np array from pandas dataframe
    train_data = train_pd.values

    HIVAE.train(args, train_data, types_dict, miss_mask, n_samples)