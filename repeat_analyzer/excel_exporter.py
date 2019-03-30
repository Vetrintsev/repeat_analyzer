#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import datetime
import xlsxwriter
from collections import Counter


# In[4]:


def series_of_tuples_to_data_frame(series, columns):
    return pd.DataFrame(
        data = list(series.values),
        index = series.index,
        columns = columns
    )

def wrap_in_list(obj):
    return obj if (type(obj) is list) else [obj]

def series_of_tuples_to_data_frame(series, columns):
    return pd.DataFrame(
        data = list(series.values),
        index = series.index,
        columns = columns
    )


# In[6]:


class Column_format:
    def __init__(
        self, 
        workbook,
        data_type = None,
        **format_props
    ):
        self._format = None
        self.workbook = workbook
        self.data_type = data_type
        self.format_properties = self.__formats[data_type or 'default'].copy()
        self.format_properties.update(format_props)
    
    def update(self, other_format):
        
        if other_format is not None:
        
            self._format = None

            self.format_properties.update(
                other_format.format_properties if type(other_format) is Column_format else other_format
            )
        
        return self
    
    def copy(self):
        return Column_format(
            self.workbook,
            **self.format_properties
        )
    
    def darken_bg_color(self, dense = 0.1):
        
        return self.update({
            'bg_color': self.mix_hex_color(
                self.format_properties['bg_color'] if 'bg_color' in self.format_properties else '#fff',
                0,
                dense
            )
        })
    
    def detect_format(self, series, skip_detected = True):
        
        if self.data_type is None or not skip_detected:
            series_type = series.dtype.type

            for types, frmt in self.__format_mapper.items():
                if series_type in types:
                    self.data_type = frmt
                    break
                    
            else:
                self.data_type = 'default'
        
            self.update(self.__formats[self.data_type])
        
        return self
    
    @property
    def format(self):
        if self._format is None:
            self._format = self.workbook.add_format(self.format_properties)
        
        return self._format
    
    __formats = {
        'default': {'border':4, 'valign':'top'},
        'datetime': {'num_format':'dd.mm.yyyy HH:MM'},
        'float': {'num_format':'0.0', 'align':'right'},
        'int': {'num_format':'#', 'align':'right'}
    }
    
    __title_format_default = {'text_wrap': True,'bold': True,'align': 'center', 'valign': 'vcenter'}
    
    __format_mapper = {
        (np.datetime64, datetime.datetime): 'datetime',
        (np.float, np.float16, np.float32, np.float64, np.float_, np.float_power): 'float',
        (np.int, np.int0, np.int16, np.int32, np.int64, np.int8, np.int_): 'int'
    }

    @classmethod
    def default_format(cls, workbook):
        return Column_format(workbook, **cls.__formats['default'])
    
    @classmethod
    def title_format(cls, workbook):
        return Column_format(workbook, **cls.__title_format_default)
    
    @classmethod
    def __color_get_rgb(cls, hex_color):
        if type(hex_color) is int:
            hex_color = hex(hex_color)[2:].zfill(6)
        
        if type(hex_color) is not str:
            raise RepeatAnalyzerException('hex_color может принадлежать только типам \'str\' или \'int\'')
            
        if hex_color.startswith('#'):
            hex_color = hex_color[1:]
        
        if hex_color.startswith('0x'):
            hex_color = hex_color[2:]
            
        if len(hex_color) == 6:
            return [int(hex_color[c:c+2], 16) for c in range(1, 6, 2)]
        
        if len(hex_color) == 3:
            return [int(c * 2, 16) for c in hex_color]
    
    @classmethod
    def mix_hex_color(cls, color, new_color, dense = 0.1):
        
        color_rgb = cls.__color_get_rgb(color)
        new_color_rgb = cls.__color_get_rgb(new_color)

        return '#' + ''.join([
            hex(c)[2:].zfill(2)
            for c in [int(c * (1 - dense) + nc * dense) for c, nc in zip(color_rgb, new_color_rgb)]
        ])


# In[379]:


class Column_producer:
    
    def __init__(
        self,
        workbook,
        func = None,
        mapper = None,
        columns = None,
        title = None,
        title_format = None,
        title_splitter = ' | ',
        value_format = None,
        value_splitter = ' | ',
        column_width = 15,
        merge = True,
        hide = False,
    ):
        """Column_producer - универсальный обработчик для выходного столбца в Excel.
        
        workbook - экземпляр xlsxwriter.workbook
        
        func - фнукция-обработчик данных, поведение зависит от columns:
        
            если columns не определен, то функции может быть только callable 
            и ей будет передан весь DataFrame, например:
                func = lambda df: df['Наименование столбца'].some_manipulations()
            
            если columns определен, то func может принимать значения:
                # строковое представление агрегатных функций Pandas
                func = 'count' # эквивалент lambda df: df[columns].agg('count')
                
                # функция обработчик серии, функции будет передан pandas.Series столбца columns
                func = lambda x: x.count() # эквивалент lambda df: func(df[columns])
                
                # Если функция не определена, то будут выведены уникальные значения columns, 
                # отсортированные в алфавитном порядке:
                func = None
                
            если df_column имеет тип tuple, то в func будет 
            передана pandas.Series с типом данных str
        
        mapper - словарь (dict), альтернатива аргументу func,
            поэтому одноврмененно func и mapper указывать нельзя.
            Обязательно должен быть передан аргумент columns.
            Значение mapper будет передан в функцию map экземпляра pandas.Series,
            эквивалентно func = lambda series: series.map(dict).
            
            mapper = {
                'value1':'new_value1',
                'value2':'new_value2',
                ...
            }
        
        columns - наименование столбца, или группы столбцов, например:
            # будут обрабатываться данные одного столбца
            columns = 'Наименование столбца 1' 
            
            # данные нескольких столбцов будут объединены через разделитель values_splitter
            columns = ('Наименование столбца 1', 'наименование столбца 2') # тип перечисления tuple
        
        title - заголовок столбца, который будет выведен в Excel. Если title = None, то название 
            будет построено из columns, и если columns имеет тип tuple, то наименования столбцов
            будут разделены через title_splitter: title_splitter.join(columns)
            title = 'Альтернативное наименование столбца 1 для вывода в Excel'
        
        title_format - словарь для построения объекта Column_format, либо экземпляр Column_format.
            Формат в итоге будет обернут в Column_format.title_format(workbook)
        
            # на основаниие переданного dict будем построен Column_format
            title_format = {
                'bg_color': '#ff0000',
                'italic': True
            }
            
            # Может быть передан и сам экземпляр Column_format
            title_format = Column_format({
                'bg_color': '#ff0000',
                'italic': True
            })
            
        title_splitter - строка-разделитель наименований столбцов, применяется только если 
            columns принадлежит типу tuple и title равен None
        
        value_format - то же самое, что и title_format, только определяет формат самих данных,
            а не заголовков. Будет обернуто в Column_format.default_format(workbook).
        
        value_splitter - то же самое, что и title_splitter, только определяет разделитель
            самих данных, в случае если columns принадлежит типу tuple.
            
        column_width - ширина столбца в файле Excel, число. 
            column_width = 25
        """
        if type(columns) not in [str, tuple, type(None)]:
            raise TypeError(f'columns принадлежит типу \'{type(columns).__name__}\'. Допустимые типы: \'str\', \'tuple\'')
        
        self.columns = columns
        self.func = Column_producer._make_func(func, mapper, columns, value_splitter)
        self.title = Column_producer._make_title(title, columns, title_splitter)
        self.value_splitter = value_splitter
        self._format = Column_format(
            workbook, 
            **(value_format if value_format is not None else {})
        )
        self._formats_even = None
        self._title_format = Column_format(
            workbook, 
            **(title_format if title_format is not None else {})
        ).update(
            Column_format.title_format(workbook)
        )
        self.column_width = column_width
        self.merge = merge
        self.hide = hide
    
    def __call__(self, df):
        """запускает функцию-обработчик
        принимает df - pandas.DataFrame
        """
        return self.func(df)
    
    def get_title_format(self):
        return self._title_format.format
    
    def get_value_format(self, even_row):
        if self._formats_even is None:            
            self._formats_even = {
                False: self._format.copy().format,
                True: self._format.copy().darken_bg_color().format
            }
        
        return self._formats_even[even_row]
    
    def detect_value_format(self, series):        
        self._format.detect_format(series)
    
    @classmethod
    def _make_func(cls, func, mapper, columns, value_splitter):
        if (func is not None) and (mapper is not None):
            raise TypeError('Нельзя одновременно передавать два аргумента func и mapper.')
        
        if columns is None:
            if func is None:
                if mapper is None:
                    raise TypeError('func и columns не определны, не возможно идентифицировать функцию-обработчик')
                
                else:
                    raise TypeError('columns не определен, не возможно идентифицировать функцию-обработчик')
                
            if not callable(func):
                raise TypeError('func не является функцией и columns не определен, не возможно идентифицировать функцию-обработчик')

            return func
        
        if mapper is not None:
            if type(mapper) is not dict:
                raise TypeError(f'mapper имеет тип \'{type(mapper)}\', допустимый тип: \'dict\'')
                
            func = lambda series: series.map(mapper)
        
        if type(columns) is str:
            get_series = lambda df: df[columns]
        
        else:
            get_series = lambda df: pd.Series(
                [
                    value_splitter.join(
                        [v for v in cols if v.lower() not in ['nan', 'nat', 'none']]
                    ) for cols 
                    in df[list(columns)].astype('str').itertuples(False)
                ], 
                index = df.index
            )
        
        if func is None:
            get_unique = lambda s: (
                s.iloc[0]
                if s.nunique() == 1 
                else '; \n'.join(sorted([
                    v 
                    for v in s.astype('str').unique() 
                    if v.lower() not in ['nan', 'nat', 'none']
                ]))
            )
            
            return lambda df: get_unique(get_series(df))
        
        if type(func) is str:
            return {
                'first': lambda df: get_series(df).iloc[0],
                'last': lambda df: get_series(df).iloc[-1],
                'first_value': lambda df: get_series(df).dropna().iloc[0],
                'last_value': lambda df: get_series(df).dropna().iloc[-1],
            }.get(
                func,
                lambda df: get_series(df).agg(func)
            )
        
        # if callable(func):
        return lambda df: func(get_series(df))
    
    @classmethod
    def _make_title(cls, title, columns, title_splitter):
        if title is None and columns is None:
            raise TypeError('title и columns не определны, невозможно определить наименование столбца для вывода в Excel')
            
        if title is None:
            if type(columns) is str:
                return columns
            
            else:
                return title_splitter.join(columns)
        
        return title
    
    @classmethod
    def Constructor(cls, workbook, instruction, default_column_params = None):
        """Генерирует экземпляр Column_producer по короткой инструкции.
        
        workbook - экземпляр xlsxwriter.workbook
        
        instruction - может принимать следующие значения:
            
            Строку - наименование столбца. Будет сгенерирован простой обаботчик, 
            который будет выводить уникальные значения столбца:
            instruction = 'Наименование столбца 1'
            
            Кортеж/перечисление - несколько наименований столбцов, значения которых
            будут соеденены через разделитель по умолчанию:
            instruction = ('Наименование столбца 1', 'Наименование столбца 2', ...)
            
            Словарь с единственной парой ключ-значение - наименования столбца(ов) и функции, допустимые варианты:
            instruction = {'Наименование столбца 1': 'count'} # эквивалент lambda df: df['Наименование столбца 1'].agg('count)
            instruction = {'Наименование столбца 1': func} # эквивалент lambda df: func(df['Наименование столбца 1'])
            instruction = { ('Наименование столбца 1', 'Наименование столбца 2', ...) : 'count'} 
            instruction = { ('Наименование столбца 1', 'Наименование столбца 2', ...) : func} 
            
            Словарь с несколькими парами ключ-значение - напрямую передается в качестве аргументов 
            конструктору Column_producer (см. описание Column_producer.__init__() ):
            instruction = {
                'title': 'Количество строк в подгруппе',
                'func': lambda df.shape[0],
                'value_format':{'bg_color': '#ff0000', 'text_wrap': True},
                'data_type': 'int', # 'float', 'datetime'
                'column_width': 30
            } # Будет вызван конструктор Column_producer(workbook, **instruction)
        """
        params = {} if (default_column_params is None) else default_column_params.copy()
        
        if type(instruction) in [str, tuple]:
            params.update({
                'columns': instruction
            })
        
        elif type(instruction) is dict:
            if len(instruction) == 1:
                columns, func = list(instruction.items())[0]
                params.update({
                    'columns': columns,
                    'func': func
                })
            
            else:
                params.update(instruction)

        else:
            raise RepeatAnalyzerException('instruction может принадлежать только одному из типов \'str\', \'tuple\' или \'dict\'')
        
        return Column_producer(workbook = workbook, **params)


# In[385]:


class Part_of_data:
    
    class DuplicatedColumnTitles(Exception): pass
    class GroupColumnsHasNans(Exception): pass
    
    _parent_layer_sorting_name = '{_parent_layer_sorting_name_}'
    _default_separator_label = '+{:.0f} ...'

    def __init__(
        self, df, workbook, sheet, 
        group_columns, export_columns, 
        sort_by = None, sort_ascending = None, 
        default_column_params = None, 
        collapse_rows_after = None, collapsed_separator_label = _default_separator_label,
        detail_data_params = None
    ):
        self.group_columns = wrap_in_list(group_columns)
        self.sort_by = sort_by
        self.sort_ascending = sort_ascending
        self.collapse_rows_after = collapse_rows_after
        self.collapsed_separator_label = collapsed_separator_label
        self.collapsed_separator_format = Column_format(workbook, bold = True)
        self.sheet = sheet
        self._even_row = True

        self._prepare_export_columns(workbook, export_columns, default_column_params)
        
        self._prepare_data(df)
        
        self._add_detail_data_layer(df, workbook, detail_data_params)
        
        self._make_iterator()
    
    def _make_iterator(self):

        if self._parent_layer_sorting_name not in self.df:
            data_iterator = self.df[self.get_column_titles()].itertuples(index = False)
            
            def iterator():
                nonlocal data_iterator
                
                for data in data_iterator:
                    yield data
                
                else:
                    data_iterator = None
                    self.iterator = None
            
        else:
            data_iterator = self.df[self.get_column_titles()].itertuples(index = True)
            cache = None
            
            def iterator():
                nonlocal cache, data_iterator
                
                if cache is not None:
                    yield cache[1:]
                    
                for data in data_iterator:
                    if cache is None:
                        cache = data
                    
                    if cache[0][0] != data[0][0]:
                        cache = data
                        break
                        
                    yield data[1:]
                
                else:
                    cache = None
                    data_iterator = None
                    self.iterator = None
        
        self.iterator = iterator
    
    def write_data(self, row, column):
        self._write_titles(row, column)
        
        return self._write_data(row + 1, column, 0) + 1, self.get_all_column_count()
    
    def _write_titles(self, row, column):
        
        for column_idx, col in enumerate(self.export_columns):
            self.sheet.set_column(column + column_idx, column + column_idx, col.column_width)
            
            self.sheet.write(
                row,
                column + column_idx,
                col.title,
                col.get_title_format()
            )
        
        if self.detail_data is not None:
            self.detail_data._write_titles(row, column + len(self.export_columns))
    
    def _write_data(self, row, column, collapse_level):
        if self.detail_data is None:
            return self._write_detail(row, column, collapse_level)

        else:
            return self._write_group(row, column, collapse_level)
    
    def _prepare_value(self, value):
        return value if ((value == value) and (value is not None)) else ''
    
    def _even_row_generator(self):
        self._even_row = not self._even_row
        return self._even_row
    
    def _write_detail(self, row, column, collapse_level):
        collapsed_row = None
        row_cursor = 0
        for data in self.iterator():
            
            if row_cursor == self.collapse_rows_after:
                collapse_level += 1
                collapsed_row = row_cursor + row - 1
            
            self._write_cells(data, row + row_cursor, column, self._even_row_generator())

            if collapse_level > 0:
                self.sheet.set_row(row + row_cursor, None, None, {'level': collapse_level, 'hidden': True})

            row_cursor += 1
        
        if collapsed_row is not None:
            self.sheet.set_row(
                collapsed_row, None, None, 
                {'level': collapse_level - 1, 'hidden': True, 'collapsed': True}
                if (collapse_level > 1) 
                else {'collapsed': True}
            )
        
        return row_cursor
    
    def _write_cells(self, data, row, column, even_row):
        for col_index, col in enumerate(self.export_columns):
            self.sheet.write(
                row,
                column + col_index,
                self._prepare_value(data[col_index]),
                col.get_value_format(even_row)
            )
    
    def _write_group(self, row, column, collapse_level):
        collapsed_row = None
        collapsed_count = 0
        columns = len(self.export_columns)
        row_cursor = 0
        
        for data in self.iterator():
            
            if row_cursor == self.collapse_rows_after:
                collapse_level += 1
                collapsed_row = row_cursor + row
                row_cursor += 1
            
            rows_height = self.detail_data._write_data(
                row + row_cursor,
                column + columns,
                collapse_level
            )
            
            self._write_grouped_cells(data, row + row_cursor, column, rows_height, self._even_row_generator())
            
            if collapsed_row is not None: 
                collapsed_count += 1
            
            row_cursor += rows_height
        
        if collapsed_row is not None:
            self._write_collapsed_separator(collapsed_row, column, collapsed_count)
            self.sheet.set_row(
                collapsed_row, None, None, 
                {'level': collapse_level - 1, 'hidden': True, 'collapsed': True}
                if (collapse_level > 1) 
                else {'collapsed': True}
            )
        
        return row_cursor
    
    def _write_collapsed_separator(self, row, column, collapsed_count):
        self.sheet.merge_range(
            row, column, 
            row, column + self.get_all_column_count() - 1,
            self.collapsed_separator_label.format(collapsed_count),
            self.collapsed_separator_format.format
        )
    
    def _write_grouped_cells(self, data, row, column, rows_height, even_row):
        for col_index, col in enumerate(self.export_columns):
            if col.merge and (rows_height > 1):
                self.sheet.merge_range(
                    row, column + col_index, 
                    row + rows_height - 1, column + col_index,
                    self._prepare_value(data[col_index]),
                    col.get_value_format(even_row)
                )

            else:
                for row_index in range(rows_height):
                    self.sheet.write(
                        row + row_index, column + col_index,
                        self._prepare_value(data[col_index]),
                        col.get_value_format(even_row)
                    )
    
    def _prepare_export_columns(self, book, columns, default_column_params = None):
        
        self.export_columns = [
            Column_producer.Constructor(book, params, default_column_params)
            for params in columns
        ]
        
        duplicate_titles = [
            str(col)
            for col, count in Counter([
                col.title 
                for col in self.export_columns
            ]).items() 
            if count > 1
        ]
        
        if len(duplicate_titles) > 0:
            raise self.DuplicatedColumnTitles(
                'Группа столбцов содержит повторяющиеся заголовки:\n"{}"'.format(
                    '"\n"'.join(duplicate_titles)
                )
            )

    def get_column_titles(self, hidden = False):
        return [col.title for col in self.export_columns if (not col.hide) or hidden]
    
    def get_column_count(self, hidden = False):
        return len([1 for col in self.export_columns if (not col.hide) or hidden])
    
    def get_all_column_count(self, hidden = False):
        return self.get_column_count(hidden) + (
            0 if (self.detail_data is None) 
            else self.detail_data.get_all_column_count(hidden)
        )

    def _add_detail_data_layer(self, df, workbook, detail_data_params):
        if detail_data_params is None: 
            self.detail_data = None

        else:
            params = detail_data_params.copy()
            params['group_columns'] = [self._parent_layer_sorting_name] + wrap_in_list(params['group_columns'])
            params['export_columns'] = [{
                'columns': self._parent_layer_sorting_name,
                'func': 'first',
                'hide': True
            }] + wrap_in_list(params['export_columns'])            

            params['sort_by'] = [self._parent_layer_sorting_name] + wrap_in_list(params.get('sort_by', []))        
    
            params['sort_ascending'] = [True] + (
                params['sort_ascending']
                if type(params.get('sort_ascending', True)) is list 
                else [params.get('sort_ascending', True)] * (len(params['sort_by']) - 1)
            )            

            self.detail_data = Part_of_data(
                df = self._prepare_detail_data(df), 
                workbook = workbook, 
                sheet = self.sheet,
                **params
            )
    
    def _prepare_data(self, df):
        
        hasnans_columns = [col for col in self.group_columns if df[col].hasnans]
        if len(hasnans_columns) > 0:
            raise self.GroupColumnsHasNans(
                'Невозможно сгруппировать данные по столбцам с пустыми значениями:\n"{}"'.format(
                    '"\n"'.join(hasnans_columns)
                )
            )
        
        def apply_data(df):
            return tuple([col(df) for col in self.export_columns])
        
        self.df = series_of_tuples_to_data_frame(
            df.groupby(
                self.group_columns,
                sort = False
            ).apply(
                apply_data
            ),
            self.get_column_titles(True)
        )
        
        self.df.index.names = [None] * len(self.df.index.names)
        
        for col in self.export_columns:
            col.detect_value_format(self.df[col.title])
        
        if self.sort_by is not None:    
            self.df.sort_values(
                self.sort_by,
                ascending = self.sort_ascending,
                inplace = True
            )
        
        self.export_columns = [col for col in self.export_columns if not col.hide]
        
    def _prepare_detail_data(self, df):
        parent_index = self.df.index

        return df.merge(
            pd.DataFrame(
                np.arange(parent_index.shape[0]), 
                index = parent_index, 
                columns = [self._parent_layer_sorting_name]
            ),
            left_on = self.group_columns,
            right_index = True,
            suffixes = ('_drop', '')
        ).drop(
            columns = [f'{self._parent_layer_sorting_name}_drop'],
            errors = 'ignore'
        )


# In[ ]:


def _get_workbook(workbook, sheet_name):
    need_close = False
    
    if type(workbook) is str:
        if not workbook.endswith('.xlsx'):
            workbook += '.xlsx'
        book = xlsxwriter.Workbook(workbook)
        need_close = True

    elif type(workbook) is xlsxwriter.Workbook:
        book = workbook

    else:
        raise TypeError(
            f'workbook принадлежит типу \'{type(workbook).__name__}\', допустим только тип \'str\' или xlsxwriter.workbook'
        )

    sheet = book.add_worksheet(str(sheet_name))
    sheet.outline_settings(True, False, True, False)
    
    return book, sheet, need_close


# In[ ]:


def _write_description(book, sheet, description, row, column, columns_width):
    if description is not None:
        sheet.merge_range(
            row, 
            column, 
            row,
            column + columns_width - 1,
            description,
            book.add_format({
                'align':'left', 'valign':'top', 'text_wrap':True
            })
        )

        sheet.set_row(
            row,
            len(description.splitlines()) * 15
        )

        return 1


# In[ ]:



def _write_data_markup(book, sheet, data_markup, row, column):
  row_cursor = 0
  
  description = data_markup.pop('description', None)
  
  if description is not None:
      description_point = row + row_cursor, column
      row_cursor += 1
  
  rows, columns = Part_of_data(
      workbook = book, sheet = sheet,
      **data_markup
  ).write_data(row + row_cursor, column)
  
  row_cursor += rows
  
  if description is not None:
      _write_description(
          book, sheet, 
          description, 
          description_point[0], description_point[1], columns
      )
  
  return row_cursor, columns


# In[ ]:


def _write_data_markups(book, sheet, data_markups, row, column):
    row_cursor = 0
    max_columns = 0
    
    for data_markup in wrap_in_list(data_markups):
        rows, columns = _write_data_markup(book, sheet, data_markup, row + row_cursor, column)
        
        row_cursor += rows + 1
        if columns > max_columns:
            max_columns = columns
        
    return row_cursor, max_columns
      


# In[ ]:


def Export_to_excel(workbook, sheet_name, data_markups, sheet_description = None):
    row = 0
    column = 0
    book, sheet, need_close = _get_workbook(workbook, sheet_name)
    
    if sheet_description is not None:
        sheet_description_point = row, column
        row += 2
    
    rows, columns = _write_data_markups(book, sheet, data_markups, row, column)
    row += rows
    
    if sheet_description is not None:
        _write_description(
            book, sheet, 
            sheet_description, 
            sheet_description_point[0], sheet_description_point[1], columns
        )
    
    if need_close:
        book.close()

