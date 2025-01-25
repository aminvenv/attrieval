import re
import numpy as np
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.exceptions import UndefinedMetricWarning
import warnings

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)


def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def is_subset(subset, main_string):
    subset_processed = preprocess(subset)
    main_string_processed = preprocess(main_string)
    return subset_processed in main_string_processed


def check_any_string(subset_list, main_string):
    for subset in subset_list:
        if is_subset(subset, main_string):
            return 1
    return 0


def evaluate_exact_match_nq(input_data):
    all_results = []
    for item in tqdm(input_data):
        result = check_any_string(item['org_answers'], item['rag_raw_answer'])
        all_results.append(result)
    final_result = np.mean(all_results)
    return final_result


def evaluate_confidence(input_data):
    relevant_author_probs = []
    nonrelevant_author_probs = []

    for row in input_data:
        row['list_of_citations_in_answer']

        probs = [np.exp(logprob) for logprob in row['logprob_of_citations_in_answer']]
        relevant_doc_index = row['bm25_top10'].index(row['new_docid'])
        for i in range(0, len(row['list_of_citations_in_answer'])):
            if row['list_of_citations_in_answer'][i] == relevant_doc_index:
                relevant_author_probs.append(probs[i])
            else:
                nonrelevant_author_probs.append(probs[i])

    return np.around(np.mean(relevant_author_probs), 4), np.around(np.mean(nonrelevant_author_probs), 4)


def evaluate_prec_recall(input_data):
    evaluation_data = list(input_data)
    pattern = r'\[\d+\]'

    for row in evaluation_data:
        text = row['rag_raw_answer']
        temp_matches = re.findall(pattern, text)
        matches = [int(item[1:-1]) for item in set(temp_matches)]
        row['llama3_citation'] = matches

    for row in evaluation_data:
        y_true = row['passages']['is_selected']
        y_pred = [1 if i in row['llama3_citation'] else 0 for i in range(0,len(y_true))]

        acc = accuracy_score(y_true, y_pred)
        pre, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')

        row['eval_results'] = {}
        row['eval_results']['all_classes'] = {'precision': pre, 'recall': rec, 'f1-score': f1, 'acc': acc}

    all_pre = []
    all_rec = []
    all_f1 = []

    for row in evaluation_data:
        all_pre.append(row['eval_results']['all_classes']['precision'])
        all_rec.append(row['eval_results']['all_classes']['recall'])
        all_f1.append(row['eval_results']['all_classes']['f1-score'])

    return {'precision': np.around(np.mean(all_pre)*100, 1), 'recall': np.around(np.mean(all_rec)*100, 1)}


def cab_evaluation(input_data_with_hlabeld_rel, input_data_with_llabeld_rel):
    pattern = r'\[\d+\]'

    input_data_with_hlabeld_rel = list(input_data_with_hlabeld_rel)
    input_data_with_llabeld_rel = list(input_data_with_llabeld_rel)

    for row_h, row_l in zip(input_data_with_hlabeld_rel, input_data_with_llabeld_rel):
        text = row_h['rag_raw_answer']
        temp_matches = re.findall(pattern, text)
        matches = [int(item[1:-1]) for item in set(temp_matches)]
        row_h['rag_citation'] = matches

        text = row_l['rag_raw_answer']
        temp_matches = re.findall(pattern, text)
        matches = [int(item[1:-1]) for item in set(temp_matches)]
        row_l['rag_citation'] = matches  

    diff_prec = []
    diff_rec = []
    for row_h, row_l in zip(input_data_with_hlabeld_rel,
                            input_data_with_llabeld_rel):

        y_true_h = row_h['passages']['is_selected']
        y_pred_h = [1 if i in row_h['rag_citation'] else 0 for i in range(0, len(y_true_h))]

        y_true_l = row_l['passages']['is_selected']
        y_pred_l = [1 if i in row_l['rag_citation'] else 0 for i in range(0, len(y_true_l))]

        acc_h = accuracy_score(y_true_h, y_pred_h)
        pre_h, rec_h, f1_h, _ = precision_recall_fscore_support(y_true_h,
                                                                y_pred_h,
                                                                average='binary')

        acc_l = accuracy_score(y_true_l, y_pred_l)
        pre_l, rec_l, f1_l, _ = precision_recall_fscore_support(y_true_l, 
                                                                y_pred_l,
                                                                average='binary')

        diff_prec.append(pre_h - pre_l)
        diff_rec.append(rec_h - rec_l)

    return np.around(np.mean(diff_prec) * 100, 2), np.around(np.mean(diff_rec) * 100, 2) 


def cas_evaluation(input_data_with_hlabeld_rel, input_data_with_llabeld_rel):
    pattern = r'\[\d+\]'

    input_data_with_hlabeld_rel = list(input_data_with_hlabeld_rel)
    input_data_with_llabeld_rel = list(input_data_with_llabeld_rel)

    for row_h, row_l in zip(input_data_with_hlabeld_rel, 
                            input_data_with_llabeld_rel):

        text = row_h['rag_raw_answer']
        temp_matches = re.findall(pattern, text)
        matches = [int(item[1:-1]) for item in set(temp_matches)]
        row_h['rag_citation'] = matches

        text = row_l['rag_raw_answer']
        temp_matches = re.findall(pattern, text)
        matches = [int(item[1:-1]) for item in set(temp_matches)]
        row_l['rag_citation'] = matches  

    diff_prec = []
    diff_rec = []
    for row_h, row_l in zip(input_data_with_hlabeld_rel, input_data_with_llabeld_rel):

        y_true_h = row_h['passages']['is_selected']
        y_pred_h = [1 if i in row_h['rag_citation'] else 0 for i in range(0, len(y_true_h))]

        y_true_l = row_l['passages']['is_selected']
        y_pred_l = [1 if i in row_l['rag_citation'] else 0 for i in range(0, len(y_true_l))]

        acc_h = accuracy_score(y_true_h, y_pred_h)
        pre_h, rec_h, f1_h, _ = precision_recall_fscore_support(y_true_h,
                                                                y_pred_h,
                                                                average='binary')

        acc_l = accuracy_score(y_true_l, y_pred_l)
        pre_l, rec_l, f1_l, _ = precision_recall_fscore_support(y_true_l,
                                                                y_pred_l,
                                                                average='binary')

        diff_prec.append(np.abs(pre_h - pre_l))
        diff_rec.append(np.abs(rec_h - rec_l))

    return np.around(np.mean(diff_prec) * 100, 2), np.around(np.mean(diff_rec) * 100, 2)