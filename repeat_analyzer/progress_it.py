#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import datetime
import time

from IPython.core.display import HTML as display_html, display, ProgressBar
from keras.callbacks import Callback


# In[ ]:


class progress_it:
    
    def __init__(self, total_count, func=None, title=''):
        """
        Usage example
        
        with progress_it(func, count, 'Title') as p:
            p(*args, **kwargs)

        """
        self._display_id = None
        self._func = func
        self.title = title
        self.total_count = total_count
    
    def __enter__(self):
        self._display = display(display_html(''), display_id=True)
        self._pbar = ProgressBar(1)
        self._pbar.html_width = '100%'
        self._pbar.display()
        
        self._counter = 0
        self._base_counter = 0
        self._start_time = time.time()
        self._prev_time = self._start_time
        
        self._update('...')
        display()
        
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        delta = datetime.timedelta(seconds=time.time() - self._start_time)
        self._counter = self.total_count
        self._update(
            '<font color="green">{}</font> <font color="grey">({})</font>'.format(
                datetime.datetime.now().strftime('%B %d %H:%M'),
                ' '.join([
                    part 
                    for part in [
                        f'{delta.days:.0f} дн. ' if delta.days>0 else None,
                        f'{(delta.seconds // 3600):.0f} ч.' if (delta.seconds // 3600) > 0 else None,
                        f'{(delta.seconds // 60):.0f} мин.' if (delta.seconds // 60) > 0 else f'{(delta.seconds):.0f} сек.'
                    ] if part is not None
                ])))
    
    def _update(self, target_date, message=None):
        self._display.update(
            display_html("""
<table width="100%"><tr><td style="text-align:left;">
    <b>{title}</b>
</td><td align="right">
    <font color="grey">{count:.0f} / {total_count:.0f}</font>&nbsp;&nbsp;&nbsp;&nbsp;{target_date}
</td></tr>{additional}</table>
            """.format(
                title=self.title,
                count=self._counter, 
                total_count=self.total_count,
                target_date=target_date,
                additional=f'<tr><td colspan="2">{message}</td></tr>' if (message is not None) else None)))
        
        self._pbar.progress = self._counter/self.total_count
        
    def base_counter_update(self, count=None, delta=None):
        if count is None:
            self._base_counter += delta or 1
            
        else:
            self._base_counter = count
        
        self._pbar.progress = self._counter/self.total_count
    
    def get_updater(self, message_prefix=None):
        def updater(count, message=None):
            if message is not None:
                message = (message_prefix or '') + message
            
            now_time = time.time()
            self._counter = self._base_counter+count

            if (now_time - self._prev_time) > 1:
                self._prev_time = now_time
                progress = self._counter/self.total_count
                
                self._update(
                    '...' if (progress == 0) else
                    datetime.datetime.fromtimestamp(
                        self._start_time + (now_time-self._start_time)/progress
                    ).strftime('%B %d %H:%M'),
                    message)
        
        return updater
    
    def __call__(self, *args, **kwargs):
        self._counter += 1
        now_time = time.time()

        if (now_time-self._prev_time) > 1:
            self._prev_time = now_time
            progress = self._counter/self.total_count
            
            self._update(
                '...' if (progress == 0) else
                datetime.datetime.fromtimestamp(
                    self._start_time + (now_time-self._start_time)/progress
                ).strftime('%B %d %H:%M'))
        
        return self._func(*args, **kwargs)


# In[ ]:


class Progress_bar_html:
    def __init__(self, update_speed, total_count, title='Progress'):
        self.start_time = time.time()
        self.prev_time = self.start_time
        self.update_speed = update_speed
        
        self.title = title
        self.subtitle = None
        self.count = 0
        self.total_count = total_count
        self.progress = 0
        
        self.display = display(display_html(''), display_id=True)
        self.pbar = ProgressBar(1)
        self.pbar.html_width = '100%'
        self.pbar.display()

        self.update_html('...')
    
    def set_total_count(self, count):
        self.total_count = count
        self.progress = self.count/self.total_count
        
        self.pbar.progress = self.progress
    
    def update_html(self, target_date, message = None):
        self.display.update(
            display_html("""
<table width="100%"><tr><td style="text-align:left;">
    <p><b>{title}</b></p>{subtitle}
</td><td align="right">
    <font color="grey">{count:.0f} / {total_count:.0f}</font>&nbsp;&nbsp;&nbsp;&nbsp;{target_date}
</td></tr>{additional}</table>
            """.format(
                title=self.title,
                count=self.count, 
                total_count=self.total_count,
                target_date=target_date,
                additional=f'<tr><td colspan="2" style="text-align:left;">{message}</td></tr>' if (message is not None) else None,
                subtitle='' if self.subtitle is None else f'<p>{self.subtitle}</p>')))
        
        self.pbar.progress = self.progress
    
    def update(self, count, message=None):
        self.count += count
        self.progress = self.count/self.total_count
        
        now = time.time()
        if self.progress == 1:
            delta = datetime.timedelta(seconds=time.time()-self.start_time)
            self.update_html(
                '<font color="green">{}</font> <font color="grey">({})</font>'.format(
                    datetime.datetime.now().strftime('%B %d %H:%M'),
                    ' '.join([
                        part 
                        for part in [
                            f'{delta.days:.0f} дн. ' if (delta.days > 0) else None,
                            f'{(delta.seconds // 3600):.0f} ч.' if (delta.seconds // 3600) > 0 else None,
                            f'{(delta.seconds // 60):.0f} мин.' if (delta.seconds // 60) > 0 else f'{(delta.seconds):.0f} сек.'
                        ] if part is not None
                    ])),
                message)
            
        elif (now-self.prev_time) > self.update_speed:
            self.prev_time = now
            
            self.update_html(
                '...' if (self.progress == 0) else
                datetime.datetime.fromtimestamp(
                    self.start_time + (now-self.start_time) / self.progress
                ).strftime('%B %d %H:%M'),
                message)

class Progress_it_keras(Callback):
    def __init__(self, title='Training progress', update_speed=1):
        self.update_speed = update_speed
        self.progbar = Progress_bar_html(self.update_speed, 0, title)
    
    def on_train_begin(self, logs=None):
        self.progbar.set_total_count(self.params['samples']*self.params['epochs'])
            
    def on_batch_end(self, batch, logs=None):
        log_values = []
        
        logs = logs or {}
        batch_size = logs.get('size', 0)        

        for k in self.params['metrics']:
            if k in logs:
                log_values.append((k, logs[k]))
        
        self.progbar.update(batch_size, ''.join([
            '<p><font color="grey"><b>{}</b>: {:.9f}</font></p>'.format(m, v)
            for m, v in log_values
        ]))

