import performance_eval as pe
import os
import csv
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.style as style
import matplotlib.gridspec as gridspec
import pandas as pd
import numpy as np
import pickle

def generate_csv(results_dir):
    results_files_list = os.listdir(results_dir)

    # keep just results obtained for MNIST
    mnist_files = [s for s in results_files_list if "MNIST" in s]
    mnist_files.sort()

    print('--------------------------')
    # remove unwanted files
    for file_name in mnist_files:
        if file_name.split('.pkl')[0][-1] == '3':
            mnist_files.remove(file_name)

    # separate MCAR from MAR
    mnist_files_mcar = [s for s in mnist_files if "MCAR" in s]
    mnist_files_mar = [s for s in mnist_files if "MAR" in s]
    mnist_files_mcar.sort()
    mnist_files_mar.sort()

    # group each run of the same method in the same sublist
    rdataset_grouped_mcar = [mnist_files_mcar[x:x + 3] for x in range(0, len(mnist_files_mcar), 3)]
    rdataset_grouped_mar = [mnist_files_mar[x:x + 3] for x in range(0, len(mnist_files_mar), 3)]

    path_original_dataset = results_dir + mnist_files[0]
    print('Original Dataset: ', path_original_dataset)
    myFile = open('/home/cazevedo/deeplearning/VAEGAN/results/results_mnist_rmse.csv', 'w')
    writer = csv.writer(myFile)
    writer.writerows([['Approach', 'Mechanism', 'MissingRatio', 'NRMSE_0', 'NRMSE_1', 'NRMSE_2', 'Mean', 'Std']])

    for reconstructed_datasets in rdataset_grouped_mar:
        reconstructed_datasets_path = [results_dir + s for s in reconstructed_datasets]
        print('Evaluating for: ', reconstructed_datasets)
        nrmses, mean_nrmses, std_nrmses = pe.evaluate_approach(path_original_dataset, reconstructed_datasets_path)

        print("NRMSEs : ", nrmses)
        print("NRMSE Mean : ", mean_nrmses)
        print("NRMSE Std : ", std_nrmses)

        description = reconstructed_datasets[0].split('.pkl')[0]
        description = description[0:len(description)-2].split('_')
        writer_notepad = []
        writer_notepad.append(description[1])
        writer_notepad.append(description[2])
        writer_notepad.append(description[3])
        writer_notepad += nrmses
        writer_notepad.append(mean_nrmses)
        writer_notepad.append(std_nrmses)

        writer.writerows([writer_notepad])

    for reconstructed_datasets in rdataset_grouped_mcar:
        reconstructed_datasets_path = [results_dir + s for s in reconstructed_datasets]
        print('Evaluating for: ', reconstructed_datasets)
        nrmses, mean_nrmses, std_nrmses = pe.evaluate_approach(path_original_dataset, reconstructed_datasets_path)

        print("NRMSEs : ", nrmses)
        print("NRMSE Mean : ", mean_nrmses)
        print("NRMSE Std : ", std_nrmses)

        description = reconstructed_datasets[0].split('.pkl')[0]
        description = description[0:len(description)-2].split('_')
        writer_notepad = []
        writer_notepad.append(description[1])
        writer_notepad.append(description[2])
        writer_notepad.append(description[3])
        writer_notepad += nrmses
        writer_notepad.append(mean_nrmses)
        writer_notepad.append(std_nrmses)

        writer.writerows([writer_notepad])

def plot_from_csv(csv_path):
    myFile = open(csv_path, 'r')
    data = pd.read_csv(myFile, sep=',')

    data_mar = data.loc[data['Mechanism']=='MAR']
    data_mcar = data.loc[data['Mechanism']=='MCAR']

    sns.set()
    # style.use('seaborn-whitegrid')

    #### ------------ MAR PLOT ------------------ ####
    plt.figure()
    sns_plot_mar = sns.lineplot(x="MissingRatio", y="Mean", hue="Approach", style="Approach",
                                markers = True, dashes = True, data = data_mar)

    # sns_plot_mar.errorbar(yerr="Std", data = data_mar)  # fmt=None to plot bars only

    sns_plot_mar.legend(markerscale=0.75)
    plt.setp(sns_plot_mar.get_legend().get_texts(), fontsize='8')  # for legend text
    plt.setp(sns_plot_mar.get_legend().get_title(), fontsize='7')  # for legend title

    plt.title('MAR')
    plt.xlabel('Missing Ratio (%)')
    plt.ylabel('RMSE Mean')

    fig = sns_plot_mar.get_figure()
    fig.savefig("/home/cazevedo/deeplearning/VAEGAN/results/mnist_rmse_mar.pdf")

    #### ------------ MCAR PLOT ------------------ ####
    plt.figure()
    sns_plot_mcar = sns.lineplot(x="MissingRatio", y="Mean", hue="Approach", style="Approach",
                                 markers = True, dashes = True, data = data_mcar)

    plt.title('MCAR')
    plt.xlabel('Missing Ratio (%)')
    plt.ylabel('RMSE Mean')

    sns_plot_mcar.legend(markerscale=0.75)
    plt.setp(sns_plot_mcar.get_legend().get_texts(), fontsize='8')  # for legend text
    plt.setp(sns_plot_mcar.get_legend().get_title(), fontsize='7')  # for legend title

    fig = sns_plot_mcar.get_figure()
    fig.savefig("/home/cazevedo/deeplearning/VAEGAN/results/mnist_rmse_mcar.pdf")

# def plot_mnist_sample(samples):
#     fig = plt.figure(figsize=(5, 5))
#     gs = gridspec.GridSpec(5, 5)
#     gs.update(wspace=0.05, hspace=0.05)
#
#     for i, sample in enumerate(samples):
#         ax = plt.subplot(gs[i])
#         plt.axis('off')
#         ax.set_xticklabels([])
#         ax.set_yticklabels([])
#         ax.set_aspect('equal')
#         plt.imshow(sample.reshape(28, 28), cmap='Greys_r')
#         # plt.imshow(sample, cmap='Greys_r')
#
#     return fig
#
# def plot_mnist(results_dir):
#     results_files_list = os.listdir(results_dir)
#
#     # keep just results obtained for MNIST
#     mnist_files = [s for s in results_files_list if "MNIST" in s]
#     mnist_files.sort()
#
#     print('--------------------------')
#     # remove unwanted files
#     for file_name in mnist_files:
#         if file_name.split('.pkl')[0][-1] == '3':
#             mnist_files.remove(file_name)
#
#     # separate MCAR from MAR
#     mnist_files_mcar = [s for s in mnist_files if "MCAR" in s]
#     mnist_files_mar = [s for s in mnist_files if "MAR" in s]
#     mnist_files_mcar.sort()
#     mnist_files_mar.sort()
#
#     # group each run of the same method in the same sublist
#     # rdataset_grouped_mcar = [mnist_files_mcar[x:x + 3] for x in range(0, len(mnist_files_mcar), 3)]
#     # rdataset_grouped_mar = [mnist_files_mar[x:x + 3] for x in range(0, len(mnist_files_mar), 3)]
#
#     path_original_dataset = results_dir + mnist_files[0]
#     print('Original Dataset: ', path_original_dataset)
#
#     original_dataset, _, _ = pickle.load(open(path_original_dataset, "rb"))
#
#     for r_path in mnist_files_mcar:
#         reconstructed_dataset = pickle.load(open(r_path, "rb"))
#
#     inc = incomplete_dataset.loc[i].values
#     rec = reconstructed_dataset.loc[i].values
#     orig = original_dataset.loc[i].values
#
#     samples = np.vstack([inc, rec, orig])
#     fig = plot_mnist_sample(samples)
#     plt.savefig('/home/cazevedo/deeplearning/VAEGAN/results/{}_sample_{}.png'.format(file_name.split('.pkl')[0],str(i)), bbox_inches='tight')
#     plt.close(fig)


if __name__ == "__main__":
    results_dir = '/mnt/nariz/cazevedo/'
    generate_csv(results_dir)
    # csv_path = '/home/cazevedo/deeplearning/VAEGAN/results/results_mnist_rmse.csv'
    # plot_from_csv(csv_path)
    # plot_mnist(results_dir)