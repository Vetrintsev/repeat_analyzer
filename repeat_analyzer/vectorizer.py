#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import datetime
import time
import sys
from collections import Counter
from math import log


# In[ ]:


import pandas as pd
import numpy as np
import umap
import hdbscan
from keras.layers import Input, Dense, LSTM, RepeatVector, TimeDistributed
from keras.models import Model, Sequential, model_from_json
from keras.preprocessing import sequence
import matplotlib.pyplot as plt


# In[ ]:


from repeat_analyzer.progress_it import progress_it, Progress_it_keras

__all__ = [
    'FeauterEncoder', 'FeauterNormalizer', 'FeauterOneHotEncoder', 
    'FeauterVectorizer', 'SequenceVectorizer', 'Clusterizer'
]


# In[ ]:


def sort_2d_array(data, sort_by=None, ascending=None):
    sort_by = (sort_by or list(range(data.shape[1])))[::-1]
    ascending = (ascending or [True] * len(sort_by))[::-1]
    
    for asc, i in enumerate(sort_by):
        data = data[
            np.argsort(data[:, i], kind='mergesort') 
            if ascending[asc] else 
            np.flip(np.argsort(data[:, i], kind='mergesort'))
        ]

    return data


# In[ ]:


class FeauterEncoder:
    
    def __init__(
            self,
            df,
            columns,
            index_column=None,
            optimizer='adam',
            loss='mse',
            epohs_per_iteration=500,
            error_step=0.001,
            attepmts=3,
            progress_callback=None):
        if type(columns) is str:
            columns = (columns,)
        
        unexpected_columns = [col for col in columns + (tuple() if index_column is None else (index_column,)) if col not in df]
        if len(unexpected_columns):
            raise AttributeError('В df отсутствуют столбцы:\n{}'.format(
                '\n'.join(unexpected_columns)
            ))
        
        self.columns = columns
        self.index_column = index_column
        self.mapper = {
            col: {k: v for v, k in enumerate(df[col].unique(), 1)}
            for col in columns
        }
        
        encoder, correction_value, correction_factor, best_error = self._fit(
            X=self._ohe_multiseries(df),
            optimizer=optimizer,
            loss=loss,
            epohs=epohs_per_iteration,
            error_step=error_step,
            attepmts=attepmts,
            progress_callback=progress_callback
        )
        
        self.model_params = {
            'architecture': encoder.to_json(),
            'weights': [arr.copy() for arr in encoder.get_weights()],
            'correction_value': correction_value,
            'correction_factor': correction_factor,
            'bad_samples': best_error
        }
    
    def __call__(self, df):
        X = self._ohe_multiseries(df)
        
        encoder = model_from_json(self.model_params['architecture'])
        encoder.set_weights(self.model_params['weights'])
        
        return (encoder.predict(X) - self.model_params['correction_value']) * self.model_params['correction_factor']
    
    def _ohe_multiseries(self, df):
        index = df.index if self.index_column is None else df[self.index_column]
        
        return np.hstack([
            self._ohe_series(self.mapper[column], df[column], index)
            for column in self.columns
        ])
    
    def _ohe_series(self, mapper, series, index):
        a = series.map(mapper).fillna(0).to_numpy(np.int32)
        ai = index.map(
            {k: v for v, k in enumerate(index.unique())}
        ).fillna(0).to_numpy(np.int32)
        
        if index.hasnans:
            sys.stderr.write('Индекс содержит пустые значения, что может привести к искажению классифкатора.\n')
        
        x = len(mapper)+1
        y = ai.max()+1
        
        b = np.zeros((y,x), dtype=np.int32)
        np.add.at(b, (ai,a), 1)

        return b.clip(max = 1)[:, 1:]
    
    def _fit(self, X, optimizer='adam', loss='mse', epohs=500, error_step=0.001, attepmts=3, progress_callback=None):
        if progress_callback is True:
            progress_callback = self._get_progress_callback()
            
        progress_it = callable(progress_callback)

        Xu, sample_weight = self._get_unique_rows(X)
        if progress_it: progress_callback(0, f'Запуск обучения: {len(Xu):.0f} уникальных сэмплов из {len(X):.0f}.')
        
        if ((X.sum(1).max()-X.sum(1).min()) / X.shape[1]) > 0.1:
            column_names = '"{}"'.format('","'.join(self.columns))
            sys.stderr.write(
f"""Слишком сложный классификатор, возможно неправильно определена группировка
для категорий: {column_names}
один из сэмплов принадлежит сразу {X.sum(1).max():.0f} класс(у/ам) из {X.shape[1]}.\n"""
            )
        
        encoder, autoencoder = self._make_model(Xu)
        autoencoder.compile(optimizer, loss)

        error, best_error, attempt, initial_epoch, convergence, reset_weights = 1, 1, 1, 0, False, False

        while True:

            autoencoder.fit(
                Xu, Xu, 
                epochs=initial_epoch + epohs,
                initial_epoch=initial_epoch,
                verbose=0,
                sample_weight=(sample_weight*error + 1*(1-error))
            )

            last_error = error
            error, bad_samples, bad_features = self._calculate_error(Xu, autoencoder.predict(Xu))

            if error > last_error:
                autoencoder.set_weights(weights)

            if (error <= error_step) or (attempt == attepmts): 
                break

            elif (last_error-error) > error_step: 
                convergence = True
                weights = [arr.copy() for arr in autoencoder.get_weights()]
                if error < last_error:
                    best_error = error
                    best_weights = [arr.copy() for arr in weights]

                initial_epoch = autoencoder.history.params['epochs']
                if progress_it: progress_callback(
                    1 - log(max(1, error // error_step), 1 // error_step), 
                    f'Идет обучение: погрешность = {error:.2%}'
                )

            elif convergence:
                attempt += 1

                if not reset_weights:
                    sample_weight = sample_weight*0 + 1
                    reset_weights = True

                weights[-2][:, bad_features] = 0
                weights[-1][bad_features] = 0

                autoencoder.set_weights(weights)

                if progress_it: progress_callback(
                    1 - log(max(1, error // error_step), 1 // error_step), 
                    f'Идет обучение: сброс коэффициентов с фич {bad_features}'
                )

                autoencoder.fit(
                    np.repeat(Xu[bad_samples], int(attempt * len(Xu) / len(bad_samples)), 0), 
                    np.repeat(Xu[bad_samples], int(attempt * len(Xu) / len(bad_samples)), 0), 
                    epochs = autoencoder.history.params['epochs'] + epohs, 
                    initial_epoch = autoencoder.history.params['epochs'],
                    verbose = 0
                )

                if progress_it: 
                    error = self._calculate_error(Xu, autoencoder.predict(Xu))[0]
                    progress_callback(
                        1 - log(max(1, error // error_step), 1 // error_step), 
                        f'Идет обучение: переобучение на сэмплах {bad_samples} до погрешности {error:.2%}'
                    )

                error = 1

        if best_error < error:
            autoencoder.set_weights(best_weights)
            error = best_error
        
        yu = encoder.predict(Xu)
        correction_value = yu.min()
        correction_factor = 1 / (yu.max()-correction_value)

        progress_callback(
            1 if (error == 0) else log(max(1, error // error_step), 1 // error_step),
            f'Обучение завершено: погрешность {error:.2%}'
        )

        return encoder, correction_value, correction_factor, best_error
    
    @classmethod
    def _make_model(cls, X):
        input_len = X.shape[1]
        output_len = int(log(input_len, 1.3))
        units_len_1 = int(input_len * (1+log(X.sum(1).max(), 2)) )
        units_len_2 = input_len

        input_shape = Input((input_len, ))

        encoder = Sequential([
            Dense(units_len_1, activation='elu'),
            Dense(units_len_1, activation='elu'),
            Dense(units_len_2, activation='elu'),
            Dense(output_len, activation='softmax')
        ])

        decoder = Sequential([
            Dense(units_len_2, activation='elu'),
            Dense(units_len_1, activation='elu'),
            Dense(units_len_1, activation='elu'),
            Dense(input_len, activation='linear')
        ])

        autoencoder = Model(input_shape, decoder(encoder(input_shape)))

        return encoder, autoencoder


    @classmethod
    def _calculate_error(cls, X, y):
        thresholds = [
            col_y[col_X != 1].max() 
            if (np.count_nonzero(col_X != 1) > 0) 
            else col_y.min() - (col_y.max()-col_y.min()) / len(col_y)
            for col_X, col_y in zip(X.T, y.T)
        ]

        y_int = (y > thresholds).astype(np.int32)

        selector = np.all(
            np.equal(
                X, 
                y_int
            ), 
            1
        )

        error = 1 - np.count_nonzero(selector) / X.shape[0]
        bad_samples = np.unique(np.argwhere(selector == False))
        bad_feauters = np.unique(np.argwhere((y_int[selector == False] != X[selector == False]).sum(0)))

        return error, bad_samples, bad_feauters
    
    @classmethod
    def _get_unique_rows(cls, data):
        """returns unique_samples, sample_weigths"""
        sorted_data =  sort_2d_array(data) # data[np.lexsort(data.T), :]
        row_mask = np.append([True], np.any(np.diff(sorted_data,axis = 0), 1))
        repeats = Counter([tuple(row) for row in sorted_data])
        data = sorted_data[row_mask]
        return data, np.array([1+log(repeats[tuple(row)]) for row in data])
    
    @classmethod
    def _get_progress_callback(cls):
        def func(progress, message):
            print(message)
        
        return func


# In[ ]:


class FeauterOneHotEncoder:
    
    def __init__(self, df, column, index_column=None):
        self.column = column
        self.index_column = index_column
        self.mapper = {k: v for v, k in enumerate(df[column].unique(), 1)}
    
    def __call__(self, df):
        index = df.index if (self.index_column is None) else df[self.index_column]
        return self._ohe_series(self.mapper, df[self.column], index)
    
    def _ohe_series(self, mapper, series, index):
        a = series.map(mapper).fillna(0).to_numpy(np.int32)
        ai = index.map(
            {k: v for v, k in enumerate(index.unique())}
        ).fillna(0).to_numpy(np.int32)
        
        if index.hasnans:
            sys.stderr.write('Индекс содержит пустые значения, что может привести к искажению классифкатора.\n')
        
        x = len(mapper)+1
        y = ai.max()+1
        
        b = np.zeros((y,x), dtype=np.int32)
        np.add.at(b, (ai,a), 1)

        return b.clip(max = 1)[:, 1:]


# In[ ]:


class FeauterNormalizer:
    
    err_nonexist_columns = 'В df отсутствуют столбцы:\n{}'
    err_unsuported_dtypes = 'Присутствуют столбцы с неподдерживаемыми типами данных:\n{}'
    
    def __init__(self, df, columns, index_column=None):
        instructions = [
            col.copy() if (type(col) is dict) else {'column': col} 
            for col in columns
        ]
        columns = [col['column'] for col in instructions]

        # проверяем входные данные
        nonexist_columns = [col for col in columns + ([] if index_column is None else [index_column]) if col not in df]
        if len(nonexist_columns):
            raise AttributeError(self.err_nonexist_columns.format('\n'.join(nonexist_columns)))
        
        unsupported_dtypes = [f'{col} ({df[col].dtype.type.__name__})' for col in columns if df[col].dtype.kind not in 'iufc']
        if len(unsupported_dtypes):
            raise TypeError(self.err_unsuported_dtypes.format('\n'.join(unsupported_dtypes)))
        
        # парсим инструкции
        for col in instructions:
            col['agg'] = col.get('agg', 'sum')
            
            col['algorithm'] = self.algorithms[
                col.get('algorithm', 'linear')
            ](
                df[col['column']],
                **col.get('params', {})
            )
        
        self.instructions = instructions
        self.index = index_column
    
    def __call__(self, df):
        df = df.groupby(
            df.index if (self.index is None) else self.index,
            sort=False
        ).agg({
            instruction['column']: instruction['agg']
            for instruction in self.instructions
        })
        
        return np.vstack([
            instruction['algorithm'](
                df[instruction['column']]
            ) for instruction in self.instructions
        ]).T.clip(min = 0, max = 1)
    
    class Linear:
        def __init__(self, series, fillna_value=0, correction_value=None, correction_factor=None):
            self.correction_value = correction_value                 or series.fillna(fillna_value).to_numpy(np.float).min()
            self.correction_factor = correction_factor                 or (1/(series.fillna(fillna_value).to_numpy(np.float).max() - self.correction_value))
            self.fillna_value = fillna_value
        
        def __call__(self, series):
            return (series.fillna(self.fillna_value).to_numpy(np.float)-self.correction_value) * self.correction_factor
    
    class Logarithm:
        def __init__(self, series, fillna_value=0, correction_value=None, logarithm_base=None):
            self.correction_value = correction_value                 or series.fillna(fillna_value).to_numpy(np.float).min()
            self.logarithm_base = logarithm_base                 or (series.fillna(fillna_value).to_numpy(np.float).max()-self.correction_value)
            self.fillna_value = fillna_value
            
        def __call__(self, series):
            values = series.fillna(self.fillna_value).to_numpy(np.float) 
            selector = values >= 0
            values[selector] = np.log(values[selector]-self.correction_value+1) / np.log(self.logarithm_base+1)
            return values
    
    algorithms = {
        'linear': Linear,
        'logarithm': Logarithm
    }


# In[ ]:


class FeauterVectorizer:
    
    def __init__(
            self, 
            df, 
            encoders=None,
            normalizers=None,
            index_column=None):
        """
        df - экземпляр pandas.DataFrame - Исходные данные, для обучения векторизаторов.
        
        encoders - список инуструкций для автокодировщиков:
            encoders = [
                
                # Наименование столбца, содержащего категории (например перечисление: цвет, версия и т.д.)
                'Наименование столбца 1',
                
                # Кортеж из наименований - оптимальное решение для кодирования тематик, и других иерархических структур
                # Порядок имеет значение! Более общий признак должен быть левее, более детального.
                # Например ('Группа тематик', 'Подгруппа тематик', 'Наименование тематики'),
                ('Наименование столбца 1', 'Наименование столбца 2', 'Наименование столбца 3'),
                
                
                # Словарь параметров, которые будут переданы инициализатору класса FeauterEncoder
                {
                    'columns': ('Наименование столбца 1', 'Наименование столбца 2'),
                    # 'columns': 'Наименование столбца 1',
                    'optimizer': 'adam',
                    'loss': 'mse',
                    'epohs_per_iteration': 500,
                    'error_step': 0.001,
                    'attepmts': 3
                },
                
                ...
            ]
        
        normalizers - список инструкций для нормализаторов:
            normalizers = [
                
                # Наименование столбца, содержащего меры (например: цена, возраст, количество и т.д.)
                'Наименование столбца 1',
                
                # Словарь параметров, которые будут переданы нормализатору:
                {
                    'column': 'Наименование столбца 1',
                    'agg': 'max', # функция, которая будет агрегировать данные в рамках одного эпизода
                    'algorithm': 'logarithm', # алгоритм нормализации 'linear' (по умолчанию) или 'logarithm'
                    'params': algorithm_params # словарь с параметрами, которые будут переданы алгоритму
                }
            ]
            
            algorithm_params - зависит от алгоритма:
                
                # для 'linear'
                algorithm_params = {
                    'fillna_value': 0, # Заполнитель для пропущенных значений, по умолчанию 0
                    'correction_value': None, # Вычитатель, по умолчанию минмиальное значение в выборке
                    'correction_factor': None # Мультипликатор, по умолчанию максимальное значение в выборке
                }
                
                # для 'logarithm'
                algorithm_params = {
                    'fillna_value': 0, # Заполнитель для пропущенных значений, по умолчанию 0
                    'correction_value': None, # Вычитатель, по умолчанию минмиальное значение в выборке
                    'logarithm_base': None # База для логарифма, по умолчанию максимальное значение в выборке + 1
                }
                
        """
        if (encoders is None) and (normalizers is None):
            raise TypeError('Не передан ни один параметр для векторизации: encoders, normalizers')
        
        self.index_column = index_column
        self.vectorizers = []
        
        if normalizers is not None:
            if type(normalizers) is not list: normalizers = [normalizers]
            
            self.vectorizers.append(
                FeauterNormalizer(df, normalizers, index_column))
            
        if encoders is not None:
            e_instructions, ohe_columns = self._prepare_encoder_instructions(df, encoders)
            
            if ohe_columns:
                self._make_ohencoders(df, ohe_columns, index_column)
            
            if e_instructions:
                self._make_encoders(df, e_instructions, index_column)
    
    def __call__(self, df):
        index = (df.index if (self.index_column is None) else df[self.index_column]).drop_duplicates()
        
        return pd.DataFrame(
            np.hstack([
                vectorizer(df)
                for vectorizer in self.vectorizers
            ]), 
            index)
    
    def _prepare_encoder_instructions(self, df, instructions):
        if type(instructions) is not list: instructions = [instructions]
        
        instructions = [
            i if type(i) is dict else {'columns': i}
            for i in instructions
        ]
        
        new_instructions = []
        for i in instructions:
            if type(i['columns']) is tuple:
                for idx in range(len(i['columns'])):
                    new_i = i.copy()
                    new_i.update({'columns': i['columns'][: idx+1]})
                    new_instructions.append(new_i)
            else:
                new_instructions.append(i)
        
        e_instructions = []
        ohe_columns = []
        for i in new_instructions:
            if isinstance(i['columns'], tuple) and len(i['columns'])>1:
                e_instructions.append(i)
            else:
                column = i['columns'][0] if isinstance(i['columns'], tuple) else i['columns']
                unique_values = df[column].nunique()
                if unique_values >= 5:
                    e_instructions.append(i)
                elif unique_values > 1:
                    ohe_columns.append(column)
                else:
                    sys.stderr.write(f'Столбец "{column}" содержит единственное значение и будет исключен из обучения.')
        
        return e_instructions, ohe_columns
    
    def _make_encoders(
            self, 
            df, 
            instructions, 
            index_column=None):
        all_errors = []
        
        with progress_it(len(instructions), title = 'Обучение автоэнкодеров') as progress:
        
            for instruction in instructions:
                    
                column_names = (instruction['columns'],) if type(instruction['columns']) is str else instruction['columns']
                
                progress_callback = progress.get_updater(
                    f"""<div>Кодирование "{'", "'.join(column_names)}":</div>"""
                )
                
                encoder = FeauterEncoder(
                    df=df,
                    index_column=index_column,
                    progress_callback=progress_callback,
                    **instruction
                )
                
                self.vectorizers.append(encoder)
                
                all_errors.append([column_names, encoder.model_params['bad_samples']])
                
                progress.base_counter_update()
        
        print('Отчет по энкодерам:')
        print('\n'.join([
            '"{}": ошибка {:.2%}'.format(
                '", "'.join(report[0]),
                report[1]
            ) for report in all_errors
        ]))
    
    def _make_ohencoders(self, df, columns, index_column=None):
        for column in columns:
            self.vectorizers.append(FeauterOneHotEncoder(df, column, index_column))


# In[ ]:


def group_ranges(groups, data):
    order = np.lexsort((data, groups))
    groups = groups[order]
    data = data[order]
    group_index = groups[1:] != groups[:-1]
    return np.vstack([
        np.unique(groups),
        data[np.append([True], group_index)],
        data[np.append(group_index, [True])]
    ]).T

def get_intersecting_ranges(groups, data):
    g = group_ranges(groups, data)
    return g[
        np.append([False], g[1:, 1] <= g[:-1, 1]) 
        | np.append(g[:-1, 2] >= g[1:, 2], [False]) 
        | np.append([False], g[1:, 1] <= g[:-1, 2])
        | np.append(g[:-1, 2] >= g[1:, 1], [False]),
        0
    ]

def count_bad_samples(X, y):
    bad_samples = np.empty(X.shape, np.bool)
    bad_feuters = []
    for i in range(X.shape[1]):
        bad_items = get_intersecting_ranges(X[:, i], y[:, i])
        bad_samples[:, i] = np.isin(X[:, i], bad_items)
        
        if bad_items.size > 0: 
            bad_feuters.append(i)
    
    return bad_samples, bad_feuters

def count_bad_sequences(X, y):
    bad_items, bad_feuters = count_bad_samples(X.reshape(-1, X.shape[-1]), y.reshape(-1, X.shape[-1]))
    bad_sequences = np.count_nonzero(np.any(np.any(bad_items, axis=1).reshape(X.shape[:2]), axis=1)) 
    return bad_sequences, bad_feuters, np.count_nonzero(bad_items)


# In[ ]:


class SequenceVectorizer:
    
    def __init__(self, data, progress_title='Fit progress', max_sequence_depth=None):
        """
        data - 2D-array with prefix columns (it will be dropped when learning): 
            - 0-indexed is a group_id, 
            - 1-indexed is a timestep_id
            
        """
        self.detect_sequence_shape(data, max_len=max_sequence_depth)
        
        self.progress_title = progress_title
        
        encoder = self.fit(data)
        
        self.model_params = {
            'architecture': encoder.to_json(),
            'weights': [arr.copy() for arr in encoder.get_weights()]
        }
    
    def __call__(self, data):
        """
        data - 2D-array with prefix columns (it will be dropped when learning): 
            - 0-indexed is a group_id, 
            - 1-indexed is a timestep_id
            
        """
        ids, X = self.pre_padding_sequences(data)
        
        encoder = model_from_json(self.model_params['architecture'])
        encoder.set_weights(self.model_params['weights'])
        
        return np.hstack([ids.reshape(-1, 1), encoder.predict(X)])
        
    def detect_sequence_shape(self, data, max_len_quentile=0.8, max_len=None):
        uniques, frequency = np.unique(data[:, 0], return_counts=True)
        
        target_len = np.quantile(
            frequency[np.searchsorted(uniques, data[:, 0])], 
            max_len_quentile, 
            interpolation='lower'
        )
        
        self.sequence_shape = target_len if (max_len is None) else max(target_len, max_len), data.shape[1] - 2
        
    def pre_padding_sequences(self, data):
        """returns (sequence_ids, padded_data)"""
        data = sort_2d_array(data, [0, 1])
        indexes = np.argwhere(np.diff(data[:, 0])).reshape(-1)+1
        
        return data[np.append([0], indexes), 0], sequence.pad_sequences(
            np.split(data[:, 2:], indexes), 
            maxlen=self.sequence_shape[0], 
            dtype=data.dtype,
            padding='post',
            truncating='post'
        )
    
    def make_model(self):
        depth, width = self.sequence_shape
        units = width*depth*4
        vector_width = int(width * (1+log(depth)))
        
        input_shape = Input(self.sequence_shape)
        
        encoder = Sequential([
            LSTM(units, activation='elu'),
            Dense(units, activation='elu'),
            Dense(vector_width, activation='softmax')
        ])
        
        decoder = Sequential([
            Dense(units, activation='elu'),
            RepeatVector(depth),
            LSTM(units, activation='elu', return_sequences=True),
            TimeDistributed(Dense(width, activation='linear'))
        ])
        
        autoencoder = Model(input_shape, decoder(encoder(input_shape)))
        
        return encoder, autoencoder
    
    def fit(self, data, epochs_per_step = 5, min_error_step=0.001, attempts=3):
        if attempts <= 0:
            raise TypeError(f'Количество попыток attempts не может быть меньше 0, передано: {attempts}')
        
        encoder, autoencoder = self.make_model()
        
        autoencoder.compile(optimizer='rmsprop', loss='mse')
        
        ids, data = self.pre_padding_sequences(data)
        
        progress = Progress_it_keras(self.progress_title)
        sequences_count = data.shape[0]
        sequences_depth = data.shape[1]
        feuters_count = data.shape[2]
        items_count = data.size
        epochs_counter = 0
        prev_error = 0
        speed = 1
        
        print(f'Запуск обучения:\nСэмплов {sequences_count}\nГлубина последовательностей {sequences_depth}\nКоличество фич {feuters_count}')
        
        while attempts > 0:
            
            autoencoder.fit(
                data, data,
                epochs=epochs_counter + epochs_per_step,
                initial_epoch=epochs_counter,
                verbose=0,
                callbacks=[progress]
            )
            
            epochs_counter += epochs_per_step
            
            bad_sequences, bad_feuters, bad_items = count_bad_sequences(data, autoencoder.predict(data))
            error = bad_sequences/sequences_count
            feauters_error = len(bad_feuters)/feuters_count
            items_error = bad_items/items_count
            
            if epochs_counter > epochs_per_step:
                speed = prev_error-error
                if speed < 0:
                    attempts -= 1
                elif ((speed < min_error_step) and (error < 0.9)) or (error <= min_error_step): 
                    break
            
            progress.progbar.subtitle = f"""
<p>Доля ошибочных предсказаний: {error:.1%} ({-speed:.1%})</p>
<p>Доля ошибочных фич: {feauters_error:.1%}</p>
<p>Доля ошибочных прогнозов: {items_error:.1%}</p>"""
            
            prev_error = error
            
            # DELETE, TEST ONLY
            attempts -= 1
            
        print(f'Доля ошибочных предсказаний: {error:.1%} ({-speed:.1%})')
        print(f'Доля ошибочных фич: {feauters_error:.1%}')
        print(f'Доля ошибочных прогнозов: {items_error:.1%}')
        
        return encoder


# In[ ]:


class Clusterizer:
    
    def __init__(self, data, verbose=True):
        """
        data - 2D-array with prefix columns (it will be dropped when learning): 
            - 0-indexed is a sample-id
        
        """
        if verbose: print('Запуск кластеризации...')
        
        self.fit_umap(data)
        self.fit_hdbscan( self.transform_umap(data) )
        
        if verbose: 
            print(f'Кластеризация завершена, выявлено {(self.hdbscan.labels_.max()+1):.0f} кластеров.')
            self.draw(data)

    def __call__(self, data):
        """
        data - 2D-array with prefix columns (it will be dropped when learning): 
            - 0-indexed is a sample-id
        
        """
        return self.transform_hdbscan( self.transform_umap(data) )
    
    def fit_umap(self, data):
        self.umap = umap.UMAP(n_neighbors = 30, n_components = 2, min_dist = 0)
        self.umap.fit(data[:, 1:])
    
    def transform_umap(self, data):
        return np.hstack([data[:, 0].reshape(-1, 1), self.umap.transform(data[:, 1:])])
    
    def fit_hdbscan(self, data):
        min_cluster_size = int(data.shape[0]*0.01)
        
        self.hdbscan = hdbscan.HDBSCAN(
            min_samples=1,
            min_cluster_size=min_cluster_size,
            prediction_data=True
        ).fit(data[:, 1:])
    
    def transform_hdbscan(self, data):
        labels, probabilities = hdbscan.approximate_predict(self.hdbscan, data[:, 1:])
        
        return np.vstack([
            data[:, 0],
            labels,
            probabilities
        ]).T
    
    def draw(self, data):
        
        points = self.transform_umap(data)
        clusters = self.transform_hdbscan(points)
        
        plt.figure(figsize=(20,10))
        plt.scatter(
            points[clusters[:, 1] >= 0, 1], 
            points[clusters[:, 1] >= 0, 2], 
            c=clusters[clusters[:, 1] >= 0, 1], 
            s=0.5, cmap='Spectral'
        )
        plt.scatter(
            points[clusters[:, 1] < 0, 1], 
            points[clusters[:, 1] < 0, 2], 
            c='#7f7f7f', 
            s=0.5
        )

