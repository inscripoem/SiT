import time
from rich.progress import Progress,TextColumn,BarColumn,TaskProgressColumn,MofNCompleteColumn
import rich.progress


for epoch in range(10):
    header = 'Epoch: [{}/{}]'.format(epoch, 10)
    with Progress(
            TextColumn(header, justify="left"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("{task.description}", justify="left"),
        ) as progress:
        task = progress.add_task("", total=len(range(100)))
        i=0
        for iteration in range(100):
            space_fmt = ':' + str(len(str(len(range(100))))) + 'd'
            log_msg = '\t'.join([
                    header,
                    '[{0' + space_fmt + '}/{1}]',
                    'eta: {eta}',
                    '{meters}',
                    'time: {time}',
                    'data: {data}'
                ])
            progress.update(
                        task, advance=1, 
                        description=log_msg.format(
                            i, len(range(100)), eta='123',
                            meters='str(self)',
                            time='str(iter_time)', data='str(data_time)')
                        )
            time.sleep(0.1)
            i+=1

