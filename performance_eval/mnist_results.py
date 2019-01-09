import performance_eval as pe
import os
import csv

def generate_csv():
    results_dir = '/mnt/nariz/cazevedo/'
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

def plot_from_csv():

if __name__ == "__main__":
    # generate_csv()
