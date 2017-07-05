"""
kfold_pipeline.py

Run multiple deep learning pipelines with k-fold cross-validation, and 
summarize discriminatory features using deepLIFT contribution scores and
paired t-tests with adjustment for multiple comparisons (Bonferroni correction).

Requires:   NumPy, SciPy RIDDLE, pipeline.py (and their dependencies)

Author:     Ji-Sung Kim, Rzhetsky Lab
Copyright:  2016, all rights reserved
"""

from __future__ import print_function

import sys; sys.dont_write_bytecode = True
import os
import time
import pickle

DATA_DIR = '_data'
CACHE_DIR ='_cache'
SEED = 109971161161043253 % 8085

import numpy as np
np.random.seed(SEED) # for reproducibility, must be before Keras imports!
from riddle import *
from pipeline import eprint, pickle_object, run_pipeline

# -------------------------- HELPER FUNCTIONS -------------------------------- #

'''
* Prints metrics out in an organized fashion.
* Expects:
    - list of loss scores (losses)
    - list of accuracies (accs)
    - list of runtimes (runtimes)
    - string identifier (id_string)
'''
def print_metrics(losses, accs, runtimes, id_string):
    print('#' * 72)
    print('for ... ' + id_string)
    print('losses = {}'.format(losses))
    print('accs   = {}'.format(accs))
    print('runtimes = {}'.format(runtimes))
    print()
    print('avg loss  = {:.5f}'.format(np.mean(losses)))
    print('avg acc   = {:.5f}'.format(np.mean(accs)))
    print('avg runtime = {:.5f} s'.format(np.mean(runtimes)))
    print('#' * 72)
    print()

''' 
* Runs multiple pipelines in line with k-fold cross-validations, using a 
  different partition as the test partition each time.
* Expects:
    - model module (model_module)
    - dictionary of model model parameters (model_params)
    - all data (X, y)
    - numbers of features (nb_features) and classes (nb_classes)
    - number of partitions for k-fold cross-validation (k)
    - boolean of whether to run model interpretation (interpret_model)
    - string out directory (out_directory)
    - string identifier (id_string)
* Returns:
    - tuple of lists (losses, accuracies)
    - tuple of lists sums of differences and sums of deepLIFT contrib scores; 
      these are used for t-tests in feature interpretation 
      (list_contrib_sums_D, list_contrib_sums_D2, list_contrib_sums)
    - pairs of compared classes (pairs)
'''
def kfold_run_pipeline(model_module, model_params, X, y, nb_features, nb_classes, 
    k, perm_indices, interpret_model, out_directory, id_string):
    losses, accs, runtimes = [], [], []
    list_contrib_sums_D, list_contrib_sums_D2, list_contrib_sums = \
        [], [], []

    print('Starting trials ... k = {}'.format(k))
    print()

    pairs = None
    for k_idx in range(k):
        print('Partition k = {}'.format(k_idx))
        print()
        data_partition_dict = emr.get_k_fold_partition(X, y, k_idx=k_idx, k=k, 
            perm_indices=perm_indices)
        sub_out_directory = '{}/{}_idx_partition'.format(out_directory, k_idx)
        if not os.path.exists(sub_out_directory): os.makedirs(sub_out_directory)

        (loss, acc, runtime), (sums_D, sums_D2, sums_contribs, curr_pairs) = \
            run_pipeline(model_module, model_params[k_idx], 
                data_partition_dict, nb_features=nb_features, 
                nb_classes=nb_classes, interpret_model=interpret_model, 
                out_directory=sub_out_directory)
        
        losses.append(loss)
        accs.append(acc)
        runtimes.append(runtime)

        list_contrib_sums_D.append(sums_D)
        list_contrib_sums_D2.append(sums_D2)
        list_contrib_sums.append(sums_contribs)

        if pairs is None: pairs = curr_pairs
        else: assert pairs == curr_pairs

        print('Finished with partition k = {}'.format(k_idx))
        print('=' * 72)
        print()


    return (losses, accs, runtimes), (list_contrib_sums_D, 
        list_contrib_sums_D2, list_contrib_sums), pairs

'''
* Totals up a list of list of sums to a list of sublist sums. 
* Expects:
    - a lists of list of sums (list_sums)
* Returns a list of sums.
'''
def compute_total_sums(list_sums):
    total_sums = list_sums[0]

    for i in range(1, len(list_sums)):
        for j in range(len(total_sums)):
            total_sums[j] = np.add(total_sums[j], list_sums[i][j])

    return total_sums

# ------------------------------- SCRIPT ------------------------------------- #

'''
* Run multiple experiment pipelines using k-fold cross-validation.
> Command line arguments:
    + id_string = string id used in output filepaths
    + data_fn = string data file name
    + interpret_model = boolean whether to compute feature importance scores
    + prop_missing = float proportion of data to randomly simulate as missing
'''
def main(args):
    k = 10 # ten partitions for k-fold cross-validation

    try: id_string = args[1]
    except: 
        id_string = 'dummy'
        eprint('Using default id_string = \'{}\''.format(id_string))

    try: data_fn = args[2]
    except: 
        data_fn = 'dummy.txt'
        eprint('Using default data_fn = \'{}\''.format(data_fn))

    try: interpret_model = args[3].lower() == 'true' or args[3].lower()[0] == 't'
    except:
        interpret_model = True
        eprint('Using default interpret_model = {}'.format(interpret_model))

    try: prop_missing = float(args[4])
    except: 
        prop_missing = 0.0
        eprint('Using default prop_missing = {}'.format(prop_missing))

    data_path = '{}/{}'.format(DATA_DIR, data_fn)
    icd9_descript_path = '{}/{}'.format(DATA_DIR, 'phewas_codes.txt')

    model_module = models.deep_mlp
    model_id = model_module.__name__.split('.')[2]
    data_name = ''.join(data_fn.split('.')[:-1])

    if not os.path.exists('out'): os.makedirs('out')
    if not os.path.exists('out/more'): os.makedirs('out/more')
    out_directory = 'out/more/{}_{}_{}'.format('riddle', data_fn, prop_missing)
    if not os.path.exists(out_directory): os.makedirs(out_directory)

    start = time.time()

    # get common data
    icd9_descript_dict = emr.get_icd9_descript_dict(icd9_descript_path) 
    X, y, idx_feat_dict, idx_class_dict = emr.get_data(path=data_path, 
        icd9_descript_dict=icd9_descript_dict, prop_missing=prop_missing)

    # print/save value-sorted dictionary of classes and features
    class_mapping = sorted(idx_class_dict.items(), key=lambda key: key[0])
    print('Class mapping:')
    print(class_mapping)
    print()
    with open(out_directory + '/class_mapping.txt', 'w+') as f:
        print(class_mapping, file=f)
    with open(out_directory + '/feature_mapping.txt', 'w+') as f:
        for idx, feat in idx_feat_dict.items():
            f.write('{}\t{}\n'.format(idx, feat))

    nb_features = len(idx_feat_dict)
    nb_classes = len(idx_class_dict)
    nb_cases = len(X)

    print('Data loaded in {:.5f} seconds'.format(time.time() - start))

    # shuffle indices and save them
    perm_indices = np.random.permutation(nb_cases)
    pickle_object(perm_indices, out_directory + '/perm_indices.pkl')
    try: # try validating shuffled indices
        with open(data_path + '_perm_indices.pkl', 'r') as f:
            exp_perm_indices = pickle.load(f)
            assert np.all(perm_indices == exp_perm_indices)
    except:
        eprint('file not found ' + data_path + '_perm_indices.pkl')
        eprint('not doing perm_indices check')

    # load saved model parameters
    model_params_fn = '{}/{}_{}_{}_param.pkl'.format(CACHE_DIR, 'riddle',
        data_fn, prop_missing)
    try:
        with open(model_params_fn, 'r') as f:
            model_params = pickle.load(f)
    except:
        eprint('Need to do parameter search!')
        eprint('Please run `parameter_search.py` with the relevant' + 
               'command line arguments')
        raise

    # run pipeline and get metric results
    (losses, accs, runtimes), (list_contrib_sums_D, 
        list_contrib_sums_D2, list_contrib_sums), pairs = \
        kfold_run_pipeline(model_module,
            model_params, X, y, nb_features=nb_features, nb_classes=nb_classes,
            k=k, perm_indices=perm_indices, interpret_model=interpret_model, 
            out_directory=out_directory, id_string=id_string)

    if interpret_model:
        total_contrib_sums_D = compute_total_sums(list_contrib_sums_D)
        total_contrib_sums_D2 = compute_total_sums(list_contrib_sums_D2)
        total_contrib_sums = compute_total_sums(list_contrib_sums)

        nb_pairs = len(pairs)

        # get descriptions of feature importance
        feat_importance_summary = feature_importance.summarize_feature_importance(
            total_contrib_sums_D, total_contrib_sums_D2, idx_feat_dict=idx_feat_dict, 
            idx_class_dict=idx_class_dict, icd9_descript_dict=icd9_descript_dict,
            pairs=pairs, nb_cases=nb_cases)

        # get frequencies of features per class
        feat_class_freq_table = frequency.get_frequency_table(X, y, 
            idx_feat_dict=idx_feat_dict, idx_class_dict=idx_class_dict)

        # get orderings
        ordering_summary = ordering.summarize_orderings(total_contrib_sums, 
            feat_class_freq_table, idx_feat_dict=idx_feat_dict, 
            idx_class_dict=idx_class_dict, icd9_descript_dict=icd9_descript_dict, 
            nb_pairs=nb_pairs)
        ordering_summary.save_individual_tables(idx_class_dict, out_directory)
        ordering_summary.save(out_directory)

    # print metrics in a pretty fashion
    print_metrics(losses, accs, runtimes, id_string=id_string)

    print('This k-fold multipipeline run script took {:.4f} seconds'
        .format(time.time() - start))

# if run as script, execute main
if __name__ == '__main__':
    import sys

    main(sys.argv)