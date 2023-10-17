import numpy as np
import os
import matplotlib.pyplot as plt
import datetime

class Data_to_monitor :
    def __init__(self, name, names) :
        self.name = name
        self.names = names
        self.num_data = len(names)
        self.reset()

    def add(self, data):
        self.data = np.concatenate([self.data, np.array(data).reshape(-1, self.num_data)])

    def mean(self):
        return np.nanmean(self.data, axis=0)

    def reset(self):
        self.data = np.zeros((0, self.num_data))

    def get_name(self):
        return self.names

    def get_best(self):
        best_idx = np.argmax(self.data)
        best = self.data[best_idx]
        return best_idx[0], best

    def get_length(self):
        return len(self.data)

    def get_data(self):
        return self.data

    def load(self, path):
        self.data = np.load(path).reshape((-1, self.num_data))

    def save(self, path):
        np.save(path, self.data)

    def plot(self, path):
        epoch = len(self.data)
        axis = np.linspace(1, epoch, epoch)
        fig = plt.figure()
        plt.title(self.name)
        for n, d in zip(self.names, self.data.T) :
            plt.plot(axis, d, label=n)
        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel(self.name)
        plt.grid(True)
        plt.savefig(path)
        plt.close(fig)

    def display(self):
        log = ['%s: %.4f'%(n, v) for n, v in zip(self.names, self.mean())]
        return ' '.join(log)
        

class Log_manager:
    def __init__(self, output_dir, class_list):
        self.dir = output_dir
        os.makedirs(self.dir, exist_ok=True)

        self.log_file = self.get_log_file()

        self.ap_names = class_list
        self.ap = Data_to_monitor('ap', self.ap_names)
        self.map = Data_to_monitor('map', ['map'])
        self.iou = Data_to_monitor('iou', ['iou'])

        self.best_map = 0
        self.best_map_epoch = 0

        '''
        if args.resume:
            self.ap.load(self.get_path('ap.npy'))
            self.map.load(self.get_path('map.npy'))
            self.iou.load(self.get_path('iou.npy'))
            
            self.best_map_epoch, self.best_map = slef.map.get_best()
        '''

    def get_path(self, *subdir):
        return os.path.join(self.dir, *subdir)

    def get_log_file(self):
        #self.log_file_name = 'log_%s.txt' % (self.args.mode)
        self.log_file_name = 'log.txt'
        open_type = 'a' if os.path.exists(self.get_path(self.log_file_name))else 'w'
        log_file = open(self.get_path(self.log_file_name), open_type)
        return log_file

    def save(self):
        self.ap.save(self.get_path('ap'))
        self.map.save(self.get_path('map'))
        self.iou.save(self.get_path('iou'))
        
        self.ap.plot(self.get_path('ap.pdf'))
        self.map.plot(self.get_path('map.pdf'))
        self.iou.plot(self.get_path('iou.pdf'))

    def add(self, data, name):
        if name == 'ap':
            self.ap.add(data)
            mAP = sum(data)/len(data)
            if(mAP > self.best_map):
                self.best_map = mAP
                self.best_map_epoch = self.map.get_length() + 1
            self.map.add(mAP)
        
        elif name == 'iou' :
            self.iou.add(data)

    def epoch_done(self):
        self.loss_every_epoch.add(self.loss_every_iter.mean())
        self.num_calssifier_pos_samples_every_epoch.add(self.num_calssifier_pos_samples_every_iter.mean())

        self.loss_every_iter.reset()
        self.num_calssifier_pos_samples_every_iter.reset()
        self.save()

    def write_log(self, log, refresh=True):
        print(log)
        self.log_file.write(log + '\n')
        if refresh:
            self.log_file.close()
            self.log_file = open(self.get_path(self.log_file_name), 'a')

    def write_cur_time(self):
        now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
        self.write_log(now)

    def done(self):
        self.log_file.close()

    def plot(self, data, label):
        epoch = len(data)
        axis = np.linspace(1, epoch, epoch)
        fig = plt.figure()
        plt.title(label)
        if(label == 'ap'):
            for ap_name, ap in zip(self.ap_names, self.ap.T) :
                plt.plot( axis, ap, label=ap_name)
        else :
            plt.plot( axis, data, label=label)
        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel(label)
        plt.grid(True)
        plt.savefig(self.get_path('%s.pdf')%(label))
        plt.close(fig)

    def get_best_map(self):
        return self.best_map

    def get_best_map_idx(self):
        return self.best_map_idx


