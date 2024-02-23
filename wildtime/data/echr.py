import re
import os
import pickle
import math
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from .utils import initialize_transform, download_detection

PREPROCESSED_FILE = 'echr.pkl'
MAX_TOKEN_LENGTH = 1024
RAW_DATA_FILE = 'echr_cases_total.pkl'
ID_HELD_OUT = 0.2
GROUP = 1


class ECHRBase(Dataset):
    def __init__(self, args):
        super().__init__()

        if args.reduced_train_prop is None:
            self.data_file = f'{str(self)}.pkl'
        else:
            self.data_file = f'{str(self)}_{args.reduced_train_prop}.pkl'
        download_detection(args.data_dir, self.data_file)
        preprocess(args)

        self.datasets = pickle.load(open(os.path.join(args.data_dir, self.data_file), 'rb'))

        self.args = args
        self.ENV = [2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015,
                    2016, 2017, 2018, 2019, 2020, 2021, 2022]
        # self.num_classes = 14 # 15 if we remove classes
        self.num_tasks = len(self.ENV)
        self.current_time = 0
        self.mini_batch_size = args.mini_batch_size
        self.task_indices = {}
        self.transform = initialize_transform(max_token_length=MAX_TOKEN_LENGTH)
        self.mode = 0

        # self.class_id_list = {i: {} for i in range(self.num_classes)}
        start_idx = 0
        self.task_idxs = {}
        self.input_dim = []
        cumulative_batch_size = 0
        for i, year in enumerate(self.ENV):
            # Store task indices
            end_idx = start_idx + len(self.datasets[year][self.mode]['labels'])
            self.task_idxs[year] = [start_idx, end_idx]
            start_idx = end_idx

            # Store class id list
            # for classid in range(self.num_classes):
            #     sel_idx = np.nonzero(np.array(self.datasets[year][self.mode]['labels']) == classid)[0]
            #     self.class_id_list[classid][year] = sel_idx
            print(f'Year {str(year)} loaded')

            # Store input dim
            num_examples = len(self.datasets[year][self.mode]['labels'])
            cumulative_batch_size += min(self.mini_batch_size, num_examples)
            if args.method in ['erm']:
                self.input_dim.append(cumulative_batch_size)
            else:
                self.input_dim.append(min(self.mini_batch_size, num_examples))

        # total_samples = 0
        # for i in self.ENV:
        #     total_samples += len(self.datasets[i][2]['category'])
        # print('total', total_samples)

    def update_historical(self, idx, data_del=False):
        time = self.ENV[idx]
        prev_time = self.ENV[idx - 1]
        self.datasets[time][self.mode]['text'] = np.concatenate(
            (self.datasets[time][self.mode]['text'], self.datasets[prev_time][self.mode]['text']), axis=0)
        self.datasets[time][self.mode]['labels'] = np.concatenate(
            (self.datasets[time][self.mode]['labels'], self.datasets[prev_time][self.mode]['labels']), axis=0)
        if data_del:
            del self.datasets[prev_time]
        # for classid in range(self.num_classes):
        #     sel_idx = np.nonzero(self.datasets[time][self.mode]['labels'] == classid)[0]
        #     self.class_id_list[classid][time] = sel_idx

    def update_historical_K(self, idx, K):
        time = self.ENV[idx]
        prev_time = self.ENV[idx - 1]
        self.window_start = self.ENV[max(0, idx - K)]
        if idx >= K:
            last_K_num_samples = self.input_dim[idx - K]
            self.datasets[time][self.mode]['text'] = np.concatenate(
                (self.datasets[time][self.mode]['text'],
                 self.datasets[prev_time][self.mode]['text'][:-last_K_num_samples]), axis=0)
            self.datasets[time][self.mode]['labels'] = np.concatenate(
                (self.datasets[time][self.mode]['labels'],
                 self.datasets[prev_time][self.mode]['labels'][:-last_K_num_samples]), axis=0)
            del self.datasets[prev_time]
            # for classid in range(self.num_classes):
            #     sel_idx = np.nonzero(self.datasets[time][self.mode]['labels'] == classid)[0]
            #     self.class_id_list[classid][time] = sel_idx
        else:
            self.update_historical(idx)

    def update_current_timestamp(self, time):
        self.current_time = time

    def __getitem__(self, index):
        pass

    def __len__(self):
        pass

    def __str__(self):
        return 'echr'


class ECHR(ECHRBase):
    def __init__(self, args):
        super().__init__(args=args)

    def __getitem__(self, index):
        if self.args.difficulty and self.mode == 0:
            # Pick a time step from all previous timesteps
            idx = self.ENV.index(self.current_time)
            window = np.arange(0, idx + 1)
            sel_time = self.ENV[np.random.choice(window)]
            start_idx, end_idx = self.task_idxs[sel_time][self.mode]

            # Pick an example in the time step
            sel_idx = np.random.choice(np.arange(start_idx, end_idx))
            index = sel_idx

        headline = self.datasets[self.current_time][self.mode]['text'][index]

        x = self.transform(text=headline)
        y = None

        return x, y

    def __len__(self):
        return len(self.datasets[self.current_time][self.mode]['labels'])


class ECHRGroup(ECHRBase):
    def __init__(self, args):
        super().__init__(args=args)
        self.num_groups = args.num_groups
        self.group_size = args.group_size
        self.window_end = self.ENV[0]
        self.train = True
        self.groupnum = 0

    def __getitem__(self, index):
        if self.mode == 0:
            np.random.seed(index)
            # Select group ID
            idx = self.ENV.index(self.current_time)
            if self.args.non_overlapping:
                possible_groupids = [i for i in range(0, max(1, idx - self.group_size + 1), self.group_size)]
                if len(possible_groupids) == 0:
                    possible_groupids = [np.random.randint(self.group_size)]
            else:
                possible_groupids = [i for i in range(max(1, idx - self.group_size + 1))]
            groupid = np.random.choice(possible_groupids)

            # Pick a time step in the sliding window
            window = np.arange(max(0, idx - groupid - self.group_size), idx + 1)
            sel_time = self.ENV[np.random.choice(window)]
            start_idx = self.task_idxs[sel_time][0]
            end_idx = self.task_idxs[sel_time][1]

            # Pick an example in the time step
            sel_idx = np.random.choice(np.arange(start_idx, end_idx))
            headline = self.datasets[self.current_time][self.mode]['text'][sel_idx]
            x = self.transform(text=headline)
            y = None
            group_tensor = torch.LongTensor([groupid])

            del groupid
            del window
            del sel_time
            del start_idx
            del end_idx
            del sel_idx

            return x, y, group_tensor

        else:
            headline = self.datasets[self.current_time][self.mode]['text'][index]

            x = self.transform(text=headline)
            y = None

            return x, y

    def group_counts(self):
        idx = self.ENV.index(self.current_time)
        return torch.LongTensor([1 for _ in range(min(self.num_groups, idx + 1))])

    def __len__(self):
        return len(self.datasets[self.current_time][self.mode]['labels'])


"""
Categories to IDs:
    {'10': 0, '11': 1, '13': 2, '14': 3, '2': 4, '3': 5, '5': 6, '6': 7, '7': 8, '8': 9, '9': 10,
    'P1-1': 11, 'P1-3': 12, 'P4-2': 13}
    
IDs to Categories:
    {0: '10', 1: '11', 2: '13', 3: '14', 4: '2', 5: '3', 6: '5', 7: '6', 8: '7', 9: '8', 10: '9',
    11: 'P1-1', 12: 'P1-3', 13: 'P4-2'}
"""


def preprocess_orig(args):
    raw_data_path = os.path.join(args.data_dir, RAW_DATA_FILE)
    if not os.path.isfile(raw_data_path):
        raise ValueError(f'{raw_data_path} is not in the data directory {args.data_dir}!')
    base_df = pd.read_pickle(raw_data_path)
    # Load data frame from json file, group by year
    base_df['year'] = pd.DatetimeIndex(base_df['judgementdate']).year
    base_df = base_df.sort_values(by=['judgementdate'])
    df_years = base_df.groupby(pd.Grouper(key='year'))
    all_dfs = [group for _, group in df_years]
    all_years = list(base_df['year'].unique())
    dfs = []
    years = []
    if GROUP > 1:
        dfs.append(pd.concat(all_dfs[:33]))
        years.append(all_years[32])
        all_dfs = all_dfs[33:]
        all_years = all_years[33:]
        for i in range(math.ceil(len(all_years) / GROUP)):
            try:
                dfs.append(pd.concat(all_dfs[GROUP * i:GROUP * i + GROUP]))
                years.append(all_years[GROUP * i + 1])
            except:
                dfs.append(pd.concat(all_dfs[GROUP * i:]))
                years.append(all_years[-1])
    else:
        dfs = [pd.concat(all_dfs[:33])] + all_dfs[33:]
        years = all_years[32:]

    # Identify class ids that appear in all years 2012 - 2018
    # categories_to_classids = {category: classid for classid, category in
    #                           enumerate(sorted(list(set([i for sublist in base_df['violated_articles'] for i in sublist]))))}
    # classids_to_categories = {v: k for k, v in categories_to_classids.items()}

    dataset = {}
    for i, year in enumerate(years):
        if i == 0:
            continue
        # Store news headlines and category labels
        dataset[year] = {}
        df_year = dfs[i - 1]
        df_year['PCR_REMAINDER_REMAINDER_CLEANED'] = df_year['PCR_REMAINDER_REMAINDER'].apply(clean_sentences)
        samples = []
        for idx in df_year.index:
            sample = divide_chunks(df_year['PCR_FACTS'][idx], df_year['PCR_REMAINDER_REMAINDER_CLEANED'][idx])
            samples += sample
        samples_train = pd.Series(samples)

        df_year = dfs[i]
        df_year['PCR_REMAINDER_REMAINDER_CLEANED'] = df_year['PCR_REMAINDER_REMAINDER'].apply(clean_sentences)
        samples = []
        for idx in df_year.index:
            sample = divide_chunks(df_year['PCR_FACTS'][idx], df_year['PCR_REMAINDER_REMAINDER_CLEANED'][idx])
            samples += sample
        samples_val = pd.Series(samples)

        dataset[year][0] = {}
        dataset[year][0]['text'] = samples_train.to_numpy()
        dataset[year][1] = {}
        dataset[year][1]['text'] = samples_val.to_numpy()
        dataset[year][2] = {}
        dataset[year][2]['text'] = samples_val.to_numpy()

    preprocessed_data_path = os.path.join(args.data_dir, PREPROCESSED_FILE)
    pickle.dump(dataset, open(preprocessed_data_path, 'wb'))


def preprocess(args):
    np.random.seed(0)
    if not os.path.isfile(os.path.join(args.data_dir, PREPROCESSED_FILE)):
        preprocess_orig(args)
    np.random.seed(args.random_seed)


def remote_between_parenthesis(input_text: str):
    result = re.sub(r'\([^)]*\)', '', input_text)
    return result


def clean_sentences(sentences):
    return [remote_between_parenthesis(x) for x in sentences]


def divide_chunks(fact, law, stride=1024):
    result = []
    chunk = []
    cur_len = 0
    max_len = 512

    # Max facts = 512 tokens
    for paragraph in reversed(fact):
        words_no = len(paragraph.split())
        # Add the sample as a list if it is longer than MAX_SEQ_LEN
        if words_no >= max_len and len(chunk) == 0:
            chunk.append(' '. join(paragraph.split()[-max_len:]))
            break
        if cur_len + words_no >= max_len:
            # The next paragraph does not fit
            chunk = [paragraph] + chunk
            break
        else:
            # Add the sample to sequence, continue and try to add more
            chunk = [paragraph] + chunk
            cur_len += words_no
            continue
    # Concat fact and law sections
    law = chunk + law
    cur_len = 0
    max_len = 1024

    for idx, p in enumerate(law):
        words_no = len(p.split())
        # Add the sample as a list if it is longer than MAX_SEQ_LEN
        if idx == 0:
            start = 0
        elif (words_no >= stride) or (cur_len + words_no >= stride):
            start = idx - 1
            pass
        else:
            cur_len += words_no
            continue

        chunk = []
        cur_len = 0
        for paragraph in law[start:]:
            words_no = len(paragraph.split())
            # Add the sample as a list if it is longer than MAX_SEQ_LEN
            if words_no >= max_len:
                result.append(chunk)
                result.append([paragraph])
                chunk = []
                cur_len = 0
                break
            if cur_len + words_no >= max_len:
                # The next paragraph does not fit in = start for next sequence
                result.append(chunk)
                chunk = []
                cur_len = 0
                break
            else:
                # Add the sample to sequence, continue and try to add more
                chunk.append(paragraph)
                cur_len += words_no
                continue
        if len(chunk) > 0:
            result.append(chunk)
        cur_len = 0
    return result
