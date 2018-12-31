from get_datasets import credit_example
from approaches import HIVAE


if __name__ == "__main__":
    credit = credit_example()
    tf_nodes, train_pd, train_data, types_dict, miss_mask, true_miss_mask, n_samples = HIVAE.test(credit)
