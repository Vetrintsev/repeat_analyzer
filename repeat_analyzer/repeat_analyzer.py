#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import datetime
import time
import xlsxwriter
import sys
import pickle
from collections import Counter


# In[ ]:


class BadDataStructure(Exception): pass
class WrongProcedure(Exception): pass


# In[400]:


class Repeat_analyzer_model:
    
    states = {k: v for v, k in enumerate([
        'init',
        'date_column',
        'group',
        'clusterize_fit',
        'clusterize',
        'to_excel'
    ])}
    
    def __init__(self):
        self.date_column = None
        self.to_excel_parameters = {}
        self.group_parameters = {}
        self.state = self.states['init']
    
    def set_date_column(self, df, date_column):
        if type(date_column) is not str:
            raise TypeError(f'date_column принадлежит типу \'{type(date_column).__name__}\', допустимо только \'str\'')
        
        if date_column not in df.columns:
            raise AttributeError(f'date_column = [\'{date_column}\'] отсутствует среди столбцов в df.columns')

        self.date_column = date_column
        
        self.state = self.states['date_column']
    
    def group_set_parameters(self, **parameters):
        self.group_parameters.update(parameters)
    
    group_column_names = [
        'episode_id',
        'incident_id',
        'recurrence_period',
        'episode_begining',
        'episode_end'
    ]
    
    def group(
        self,
        df,
        **parameters
    ):        
        if self.state < self.states['date_column']:
            raise WrongProcedure('Метод group можно вызвать только после set_date_column')

        self.group_set_parameters(**parameters)
        
        episodes_group_by = self.group_parameters['episodes_group_by']
        incidents_group_by = self.group_parameters['incidents_group_by']
        recurrence_period = self.group_parameters['recurrence_period']
        
        if type(recurrence_period) is int:
            recurrence_period = datetime.timedelta(days = recurrence_period)
            
        elif type(recurrence_period) is dict:
            recurrence_period = datetime.timedelta(**recurrence_period)
            
        elif type(recurrence_period) is not datetime.timedelta:
            raise TypeError(f'recurrence_period принадлежит типу \'{type(recurrence_period).__name__}\', допускается только \'int\', \'dict\', \'timedelta\'')
        
        df = df.drop(
            columns = self.group_column_names + self.cluster_column_names, 
            errors = 'ignore'
        )
        
        # создадим карту групп для экономии памяти
        mapper = df.groupby(
            episodes_group_by + incidents_group_by,
            as_index = False
        ).agg({
            self.date_column: 'min'
        }).reset_index().rename(
            columns = {'index': 'incident_id'}
        ).merge(
            df[episodes_group_by].drop_duplicates().reset_index(
                drop = True
            ).reset_index().rename(
                columns = {'index': 'group_id'}
            ),
            on = episodes_group_by
        ).merge(
            df[
                episodes_group_by + incidents_group_by
            ].reset_index(),
            on = episodes_group_by + incidents_group_by
        ).drop(
            columns = episodes_group_by + incidents_group_by
        )
        
        # продолжим работу с инцидентами
        incidents = mapper.drop(
            columns = 'data_id'
        ).drop_duplicates().sort_values(
            ['group_id', self.date_column]
        ).reset_index(
            drop = True
        ).reset_index()
        
        # привяжим к инцидентам предшествующие инциденты
        incidents['prev_index'] = incidents['index'] - 1
        incidents = incidents.merge(
            incidents,
            how = 'left',
            left_on = 'prev_index',
            right_on = 'index',
            suffixes=('', '_prev')
        )
        
        # рассчитаем время, которое прошло с предидущего инцидента
        incidents['recurrence_period'] = incidents[self.date_column] - incidents[f'{self.date_column}_prev']
        
        incidents.loc[
            (incidents['recurrence_period'] > recurrence_period)
            | (incidents['group_id'] != incidents['group_id_prev']),            
            'recurrence_period'
        ] = None
        
        # определим инциденты, с которых начинаются эпизоды
        incidents.loc[
            incidents['recurrence_period'].isna(), 
            'episode_id'
        ] = incidents.loc[
            incidents['recurrence_period'].isna(), 
            'incident_id'
        ]
        
        # привяжем повторные инциденты к эпизоду
        incidents['episode_id'] = incidents['episode_id'].ffill()
        
        # Выявляем полноценные эпизоды
        incidents['episode_begining'] = ~incidents['episode_id'].isin(
            incidents[
                incidents[self.date_column] < (mapper[self.date_column].min() + recurrence_period)
            ]['episode_id'].unique()
        )
        
        incidents['episode_end'] = ~incidents['episode_id'].isin(
            incidents[
                incidents[self.date_column] > (mapper[self.date_column].max() - recurrence_period)
            ]['episode_id'].unique()
        )
        
        self.state = self.states['group']
        
        # вернем индексы и привяжем признаки к исходному DataFrame
        return df.merge(
            mapper[[
                'data_id', 
                'incident_id'
            ]].merge(
                incidents[[
                    'episode_id',
                    'incident_id',
                    'recurrence_period',
                    'episode_begining',
                    'episode_end'
                ]],
                on = 'incident_id'
            ).set_index(
                'data_id'
            ),
            left_index = True,
            right_index = True
        )
    
    def clusterize_fit(
        self, 
        df, 
        categories, 
        measures, 
        drop_incomplete_episodes = True,
        max_episode_depth = None
    ):
        if self.state < self.states['group']:
            raise WrongProcedure('Метод clusterize_fit можно вызвать только после group')
        
        from vectorizer import Feauter_vectorizer, Sequence_vectorizer, Clusterizer
        
        if drop_incomplete_episodes: 
            df = df[df['episode_begining'] & df['episode_end']]
        
        self.incidents_vectorizer = Feauter_vectorizer(
            df,
            categories,
            [{
                'column': 'recurrence_period',
                'agg': 'first', 
                'algorithm': 'logarithm'
            }] + measures,
            'incident_id'
        )
        
        self.episodes_vectorizer = Sequence_vectorizer(
            self._clusterize_prepare_incidents(df), 
            'Обучение автоэнкодера эпизодов',
            max_episode_depth
        )
        
        self.clusterizator = Clusterizer(
            self.episodes_vectorizer(incidents), 
            True
        )
        
        self.statistics = None
        
        self.state = self.states['clusterize_fit']
    
    def _clusterize_prepare_incidents(self, df):        
        incidents = self.incidents_vectorizer(df)
        
        return incidents.merge(
            df[['incident_id', 'episode_id']].drop_duplicates().set_index('incident_id'),
            left_index = True,
            right_index = True
        ).reset_index()[
            ['episode_id', 'incident_id'] + [
                col for col in incidents.columns 
                if col not in ['incident_id', 'episode_id']
            ]
        ].to_numpy(np.float32)
    
    def _clusterize_update_statistics(self, clusters):
        
        current_statistics = clusters.groupby(
            'cluster_id'
        ).agg({
            'episode_id':'count'
        }).rename(
            columns = {'episode_id': datetime.datetime.now().strftime('%Y.%m.%d %H:%M')}
        )
        
        if self.statistics is None:
            self.statistics = current_statistics
        
        else:
            self.statistics = self.statistics.merge(
                current_statistics,
                left_index = True,
                right_index = True,
            )
    
    cluster_column_names = ['cluster_id', 'clustering_quality']
    
    def clusterize(self, df):
        
        if self.state < self.states['clusterize_fit']:
            raise WrongProcedure('Метод clusterize можно вызвать только после clusterize_fit')
        
        clusters = pd.DataFrame(
            self.clusterizator(
                self.episodes_vectorizer(
                    self._clusterize_prepare_incidents(df)
                )
            ),
            columns = ['episode_id'] + self.cluster_column_names
        )
        
        self._clusterize_update_statistics(clusters)
        
        self.state = self.states['clusterize']
        
        return df.drop(
            columns = self.cluster_column_names, 
            errors = 'ignore'
        ).merge(
            clusters,
            on = ['episode_id']
        )
    
    def to_excel_set_parameters(self, **parameters):
        self.to_excel_parameters.update(parameters)
    
    def to_excel(
        self,
        df,
        workbook,
        sheet_name = 'Sheet 1',
        **parameters
    ):
        if self.state < self.states['clusterize']:
            raise WrongProcedure('Метод to_excel можно вызвать только после clusterize')
        
        from excel_exporter import Export_to_excel
        
        self.to_excel_set_parameters(**parameters)
        
        system_titles = {
            
            # Для кластеров
            'cluster_id': '№ кластера',
            'episodes_count': 'Количество эпизодов',
            
            # Для эпизодов
            'episode_id': '№ эпизода',
            'clustering_quality': 'Точность определения кластера',
            'episodes_collapsed_separator': 'еще эпизодов: {:.0f}...',
            'episode_begining': 'Найден первый инцидент',
            
            # Для инцидентов
            'incidents_count': 'Количество инцидентов',
            'incidents_collapsed_separator': 'еще инцидентов: {:.0f}...',
            'recurrence_period': 'Прошло времени до повторения'
        }
        
        if self.to_excel_parameters.get('system_column_headers', None) is not None:
            system_titles.update(self.to_excel_parameters['system_column_headers'])

        recurrence_period_unit_func = (lambda x: lambda s: s.iloc[0].total_seconds() / x)({
            'seconds': 1, 'minutes': 60, 'hours': 3600, 'days': 86400
        }[self.to_excel_parameters.get('recurrence_period_unit', 'hours')])
        
        incidents_params = {
            'group_columns': 'incident_id', 
            'export_columns': [
                self.date_column,
                {
                    'columns': 'recurrence_period', 
                    'func': recurrence_period_unit_func, 
                    'title': system_titles['recurrence_period']
                }
            ] + self.to_excel_parameters['incident_columns'],
            'sort_by': self.date_column, 'sort_ascending': True, 
            'default_column_params': self.to_excel_parameters.get('system_column_headers'),
            'collapse_rows_after': self.to_excel_parameters.get('max_episode_length', 3)
        }
        
        episodes_params = {
            'group_columns':'episode_id', 
            'export_columns':[
                {'columns': 'episode_id', 'title': system_titles['episode_id']},
                {'columns': 'clustering_quality', 'title': system_titles['clustering_quality']},
                {'columns': 'incident_id', 'func': 'nunique', 'title': system_titles['incidents_count']},
                {'columns': 'episode_begining', 'mapper':{True:'Да', False:'Нет'},'title': system_titles['episode_begining']},
            ] + self.to_excel_parameters['episode_columns'],
            'sort_by': [system_titles['clustering_quality'], system_titles['incidents_count']],
            'sort_ascending': False, 
            'default_column_params': self.to_excel_parameters.get('system_column_headers'),
            'collapse_rows_after': 10, 'collapsed_separator_label': system_titles['episodes_collapsed_separator'],
            'detail_data_params': incidents_params
        }
        
        clusters_params = {
            'df': self.df[
                self.df['episode_id'].isin(
                    self.df.groupby('episode_id').agg(
                        {'incident_id':'nunique'}
                    ).query('incident_id >= {}'.format(
                        self.to_excel_parameters.get('min_episode_length', 1)
                    )).index
                )
            ],
            'group_columns': 'cluster_id', 
            'export_columns': [
                {'columns': 'cluster_id', 'title': system_titles['cluster_id']},
                {'columns': 'episode_id', 'func': 'nunique', 'title': system_titles['episodes_count']}
            ] + self.to_excel_parameters.get('cluster_columns', []), 
            'sort_by': system_titles['episodes_count'], 'sort_ascending': False, 
            'default_column_params': self.to_excel_parameters.get('system_column_headers'),
            'detail_data_params': episodes_params
        }
        
        statistics = {
            'df': self.statistics.reset_index(),
            'group_columns': 'cluster_id',
            'export_columns': [
                {'columns': 'cluster_id', 'title': system_titles['cluster_id']}
            ] + self.statistics.columns.tolist()
        }
        
        Export_to_excel(
            workbook = workbook, 
            sheet_name = sheet_name, 
            data_markups = [
                statistics,
                clusters_params
            ], 
            sheet_description = self.to_excel_parameters.get('description')
        )
        
        self.state = self.states['to_excel']


# In[401]:


import 

class Repeat_analyzer:
    
    def __init__(self, df, date_column):
        """
        df - экземпляр pandas.DataFrame.
            В заголовках столбцов может быть только один уровень (недопускается мультииндекс)
        
        date_column - наименование столбца, который содержит даты целевых инцидентов.
            Важно, если при группировке эпизодов, в рамках одного уникального
            идентификатора инцидента будет несколько значений date_column, 
            выбрана будет наименьшая.
        """ 
        if df.columns.nlevels != 1:
            raise BadDataStructure(f'df.columns.nlevels = {df.columns.nlevels}, допустимы только одноуровневые заголовки.')

        self.model = Repeat_analyzer_model()
        self.model.set_date_column(df, date_column)
        
        self.df = df[~df[date_column].isna()].reset_index()
        self.df.index.names = ['data_id']
    
    def group(
        self,
        episodes_group_by,
        incidents_group_by,
        recurrence_period = 31
    ):
        """
        episodes_group_by - список наименований столбцов, по которым необходимо 
            сгруппировать эпизоды:
            episodes_group_by = [
                'Наименование столбца 1',
                'Наименование столбца 2',
                ...
            ]
        
        incidents_group_by - список наименований столбцов, по которым необходимо 
            сгруппировать инциденты:
            incidents_group_by = [
                'Наименование столбца 1',
                'Наименование столбца 2',
                ...
            ]
            
        recurrence_period - период, в течение которого следующий инцидент будет 
            считаться повторным:
            
            # целое число - будет интерпретировано как количество дней:
            recurrence_period = 7 # т.е. 7 дней или timedelta(days = 7)
            
            
            # словарь - будет передан функции datetime.timedelta
            recurrence_period = {'hours': 24} # т.е. 24 часа или timedelta(**recurrence_period)
            
            # экземпляр datetime.timedelta
            recurrence_period = timedelta(days = 7)
            
        """
        
        self.df = self.model.group(
            df = self.df,
            episodes_group_by = episodes_group_by,
            incidents_group_by = incidents_group_by,
            recurrence_period = recurrence_period
        )
        
        return self
    
    def clusterize(
        self,
        categories, 
        measures, 
        drop_incomplete_episodes = True,
        max_episode_depth = None
    ):
        """
        categories - список инструкций для кодирования категорий:
            Например, тематика, цвет, класс обслуживания и т.д.
            categories = [
                
                # Наименование столбца, содержащего категории (например перечисление: цвет, версия и т.д.)
                'Наименование столбца 1',
                
                # Кортеж из наименований - оптимальное решение для кодирования тематик, и других иерархических структур
                # Порядок имеет значение! Более общий признак должен быть левее, более детального.
                # Например ('Группа тематик', 'Подгруппа тематик', 'Наименование тематики'),
                ('Наименование столбца 1', 'Наименование столбца 2', 'Наименование столбца 3'),
                
                # Словарь параметров, которые будут переданы инициализатору класса Feauter_encoder
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
        
        measures - список инструкций для нормализаторов измеряемых параметров. 
            Например, цена, количество, возраст и т.д..
            
            measures = [
                
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
                    fillna_value: 0, # Заполнитель для пропущенных значений, по умолчанию 0
                    correction_value: None, # Вычитатель, по умолчанию минмиальное значение в выборке
                    correction_factor: None # Мультипликатор, по умолчанию максимальное значение в выборке
                }
                
                # для 'logarithm'
                algorithm_params = {
                    fillna_value: 0, # Заполнитель для пропущенных значений, по умолчанию 0
                    correction_value: None, # Вычитатель, по умолчанию минмиальное значение в выборке
                    logarithm_base: None # База для логарифма, по умолчанию максимальное значение в выборке + 1
                }
        
        drop_incomplete_episodes - True (по умолчанию) или False, указывает алгоритму - отбросить 
            неполные эпизоды (рекомендуется) или нет.
        
        max_episode_depth - целое число, по умолчанию None. Определяет максимальную глубину эпизодов 
            при обучении автоэнкодеров. 
            Актуально когда группировка привела к опявлению большого количества эпизиодов с большим 
            количеством инцидентов, такие ситуации могут привести к недостатоку памяти.
            По умолчанию алгоритм отбрасывает аномально большие эпизоды при обучении.
        """
        
        self.model.clusterize_fit(
            self.df,
            categories, 
            measures, 
            drop_incomplete_episodes = drop_incomplete_episodes,
            max_episode_depth = max_episode_depth
        )
        
        self.df = self.model.clusterize(self.df)
        
        return self
        
    def to_excel(
        self,
        cluster_columns,
        episode_columns,
        incident_columns,
        workbook,
        sheet_name = 'Sheet 1',
        description = None,
        max_episode_length = 5,
        min_episode_length = 2,
        recurrence_period_unit = 'hours',
        default_format = {'column_width': 15},
        system_column_headers = None
    ):
        """
        episode_columns - список столбцов вывода в Excel группирующих эпизод, 
            может принимать значения:
            
            episode_columns = [
                
                # строка - в объединенную ячейку будут выведены все уникальные 
                # значения из соответствующего столбца исходного набора данных
                'Наименование столбца',
                
                # кортеж (tuple) - в объединенную ячейку будут выведены все уникальные 
                # сочетания значений из соответствующих столбцов исходного набора данных
                ('Наименование столбца 1', 'Наименование столбца 2'),
                
                # словарь (dict) с одним ключем - в объединенную ячейку будут выведены 
                # результаты расчета функции присвоенной соответствующему ключу
                {'Наименование столбца': 'count'}, # будут рассчитано количество 
                # по столбцу 'Наименование столбца'
                
                {'Наименование столбца': lambda x: x.sum()}, # будет рассчитана сумма 
                # по столбцу 'Наименование столбца', в функцию lambda будет передан 
                # экземпляр Pandas.Series столбца 'Наименование столбца'
                
                # в качестве наименования столбца можно также передать кортеж (tuple)
                {('Наименование столбца 1', 'Наименование столбца 2'): 'count'}
                
                # словарь (dict) с несколькими ключами - данные параметры будут переданы 
                # конструктору excel_exporter.Column_producer, позволяет гибко настраивать вид и
                # содержание столбцов 
                {
                    'title': 'Количество строк в подгруппе', # в excel будет иметь такое название
                    'func': lambda df.shape[0], # выводит количество строк в выборке
                    'value_format':{'bg_color': '#ff0000', 'text_wrap': True}, # задает фон ячейке
                    'data_type': 'int', # 'float', 'datetime' - задачет тип отображения данных в excel
                    'column_width': 30 # задает ширину столбца
                }
            ]
            
        incident_columns - список столбцов вывода в Excel группирующих инциденты в 
            рамках одного эпизода. Может принимать те же значения, что и episode_columns

        workbook - определяет файл excel, в который будет записан результат.
            может принимать:

            # строку - определяет путь и наменование файла. 
            # Если расширение файла не ".xlsx", оно будет добавлено автоматически
            workbook = 'excel_file_name.xlsx'

            # может принимать экземпляр xlsxwriter.workbook
            workbook = xlsxwriter.Workbook('some file')

        sheet_name - наименование нового листа в книге excel,
            на который будет записан результат.
            sheet_name = 'Лист 1'

        description - произволдьное описание, которое будет вставлено перед шапкой 
            результирующей таблицы
            description = "Некоторое описание таблицы"

        max_episode_length - число, максимальная глубина вывода инцидентов эпизода.
            вывод инцидентов, которые глубже в иерархии, 
            зависит от аргумента group_overlimit_incidents
            max_episode_length = 5

        min_episode_length - число, минимальная глубина эпизодов.
            Эпизоды с меньшим количеством эпизодов будут
            
        recurrence_period_unit - строка, единица измерения времени между 
            повторными инцидентами.
            Возможные значения: 'seconds', 'minutes', 'hours', 'days'

        default_format - словарь, параметры формата по умолчанию.
            # задается ширина столбцов в excel по умолчанию.
            default_format = {'column_width': 15}
        
        system_column_headers - меппинг наименований системных столбцов, 
            которые будут выведены дополнительно.
            Достаточно указать наименования только по определенным столбцам, например:
            
            # переименованы будут только столбцы 'cluster_id_column_name' и 'episode_id_column_name'
            # остальные столбцы примут значение по умолчанию
            system_column_headers = {
                'cluster_id_column_name': 'Идентификатор кластера',
                'episode_id_column_name': 'Идентификатор эпизода',
            }
            
            Значения system_column_headers по умолчанию:
            {

                # Для кластеров
                'cluster_id': '№ кластера',
                'episodes_count': 'Количество эпизодов',

                # Для эпизодов
                'episode_id': '№ эпизода',
                'clustering_quality': 'Точность определения кластера',
                'episodes_collapsed_separator': 'еще эпизодов: {:.0f}...',
                'episode_begining': 'Найден первый инцидент',

                # Для инцидентов
                'incidents_count': 'Количество инцидентов',
                'incidents_collapsed_separator': 'еще инцидентов: {:.0f}...',
                'recurrence_period': 'Прошло времени до повторения'
            }
        """
        
        self.model.to_excel(
            df = self.df,
            workbook = workbook,
            sheet_name = sheet_name,
            cluster_columns = cluster_columns,
            episode_columns = episode_columns,
            incident_columns = incident_columns,
            description = description,
            max_episode_length = max_episode_length,
            min_episode_length = min_episode_length,
            recurrence_period_unit = recurrence_period_unit,
            default_format = default_format,
            system_column_headers = system_column_headers
        )
    
    def transform(self, workbook = None, sheet_name = None):
        """
        Если workbook передан и не None, то сразу будет вызван экспорт 
        в Excel с сохаренными ранее параметрами.
        
        workbook - определяет файл excel, в который будет записан результат.
            может принимать:

            # строку - определяет путь и наменование файла. 
            # Если расширение файла не ".xlsx", оно будет добавлено автоматически
            workbook = 'excel_file_name.xlsx'

            # может принимать экземпляр xlsxwriter.workbook
            workbook = xlsxwriter.Workbook('some file')

        sheet_name - наименование нового листа в книге excel,
            на который будет записан результат.
            sheet_name = 'Лист 1'
        """
        self.df = self.model.clusterize(
            df = self.model.group(
                df = self.df
            )
        )
        
        if workbook is not None:
            self.model.to_excel(
                df = self.df,
                workbook = workbook,
                sheet_name = sheet_name,
            )
    
    def save_model(self, path):
        """
        path - путь и имя файла для сохранения модели.
        """
        
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)
        
        print(f'Модель успешно сохранена в файле: {path}')
    
    @classmethod
    def load_model(cls, path, df):
        """
        path - путь к файлу для загрузки модели.
        
        df - экземпляр pandas.DataFrame. 
            Структура исходных данных должна быть схожа с первичными данными,
            на основе которых была построена модель.
        """
        self = cls.__new__(cls)
        
        with open(path, 'rb') as f:
            self.model = pickle.load(f)
        
        if df[self.model.date_column].hasnans:
            raise AttributeError('Столбец с датами содержит пустые значения, это недопустимо.')
        
        self.df = df.reset_index()
        self.df.index.names = ['data_id']

