import os
import numpy as np
import struct as st
import obspy as obs


class FrameSet(list):

    def __init__(self, version, channel_mask, station_id, acq_time, freq):
        self.version = version
        self.channel_mask = channel_mask
        self.station_id = station_id
        self.acq_time = acq_time
        self.freq = freq
        pass

    def __add__(self, other):
        new = FrameSet(self.version, self.channel_mask, self.station_id, self.acq_time, self.freq)
        for i in self:
            new.append(i)
        for i in other:
            new.append(i)
        return new

    def __repr__(self):
        return f'FrameSet(amount:{len(self)}, station_id:{self.station_id}, freq:{self.freq})'

    def info(self):
        print('version:', self.version)
        print('channel mask:', self.channel_mask)
        print('station id:', self.station_id)
        print('acq time:', self.acq_time)
        print('freq:', self.freq)
        print('number of frames:', len(self))


class Frame(list):

    def __init__(self, gps_marks):
        self.gps_marks = gps_marks

    def __repr__(self):
        return f'Frame(channels:{len(self)}, gps_marks:{len(self.gps_marks)})'


class GpsMark:

    def __init__(self, date, valid, time_acc, fix_type,
                 status_flags, additional_flags, lat, lon,
                 height_el, height_sea):
        self.date = date
        self.valid = valid
        self.time_acc = time_acc
        self.fix_type = fix_type
        self.status_flags = status_flags
        self.additional_flags = additional_flags
        self.lat = lat
        self.lon = lon
        self.height_el = height_el
        self.height_sea = height_sea

    def __repr__(self):
        if self.valid == 55:
            return f"{self.date[0]}-{self.date[1]}-{self.date[2]}T{self.date[3]}:{self.date[4]}:{self.date[5]}"
        else:
            return 'NonValid'

    def info(self):
        print('date', self.date)
        print('valid', self.valid)
        print('time_acc', self.time_acc)
        print('fix_type', self.fix_type)
        print('status_flags', self.status_flags)
        print('additional_flags', self.additional_flags)
        print('lat', self.lat)
        print('lon', self.lon)
        print('height_el', self.height_el)
        print('height_sea', self.height_sea)


def channel(num):
    channels = 0
    for i in range(3):
        channels += num >> i & 1
    return channels


def read(path, filename):

    head_len = 128
    frame_head_len = 744
    part_len = 24000
    end_head_len = 128
    disc = 6000

    data_set = os.path.join(path, filename)
    file_size = os.stat(data_set).st_size

    with open(data_set, 'rb') as f:

        version = st.unpack('9s', f.read(9))[0]
        channel_mask = channel(st.unpack('b', f.read(1))[0])
        station_id = st.unpack('<h', f.read(2))[0]
        acq_time = st.unpack('<h', f.read(2))[0]
        freq = st.unpack('<h', f.read(2))[0]
        f.read(112)

        file_set = FrameSet(version, channel_mask, station_id, acq_time, freq)

        number_of_frames = int((file_size - head_len - end_head_len)/(frame_head_len + 3*part_len))
        for i in range(number_of_frames):

            gps = []
            for j in range(int(disc/freq)):

                year = st.unpack('h', f.read(2))[0]
                month = st.unpack('b', f.read(1))[0]
                day = st.unpack('b', f.read(1))[0]
                hour = st.unpack('b', f.read(1))[0]
                minutes = st.unpack('b', f.read(1))[0]
                sec = st.unpack('b', f.read(1))[0]
                date = (year, month, day, hour, minutes, sec)
                valid_flags = st.unpack('b', f.read(1))[0]
                time_accuracy = st.unpack('i', f.read(4))[0]
                fix_type = st.unpack('b', f.read(1))[0]
                fix_flags = st.unpack('b', f.read(1))[0]
                additional_flags = st.unpack('b', f.read(1))[0]
                lon = st.unpack('i', f.read(4))[0]
                lat = st.unpack('i', f.read(4))[0]
                height_el = st.unpack('i', f.read(4))[0]
                height_sea = st.unpack('i', f.read(4))[0]

                gps.append(GpsMark(date, valid_flags, time_accuracy,
                                    fix_type, fix_flags, additional_flags,
                                    lon, lat, height_el, height_sea))

            f.read(frame_head_len - int(disc/freq)*31)

            run_frame = Frame(gps)
            for j in range(3):
                run_frame.append(np.fromfile(f, dtype='<i', count=disc))

            file_set.append(run_frame)

        end_head = f.read(end_head_len)

    return file_set


def read_hour(in_path, out_path):

    file_names = os.listdir(in_path)
    stream = obs.Stream()

    ch1 = np.array([], dtype='int32')
    ch2 = np.array([], dtype='int32')
    ch3 = np.array([], dtype='int32')

    stats_file = read(in_path, file_names[0])

    for i in stats_file[0].gps_marks:
        if str(i) != 'NonValid':
            time = i.date
            break
    utc_time = obs.UTCDateTime(time[0], time[1], time[2],
                               time[3], time[4], time[5])

    stats = {'station': 'K106'+'-'+str(stats_file.station_id),
             'network': 'T1',
             'location': 'XX',
             'channel': 0,
             'npts': 0,
             'sampling_rate': stats_file.freq,
             'starttime': utc_time,
             'mseed': {'dataquality': 'R'}}

    for name in file_names:
        file = read(in_path, name)
        for i in file:
            ch1 = np.concatenate((ch1, i[0]), dtype='int32')
            ch2 = np.concatenate((ch2, i[1]), dtype='int32')
            ch3 = np.concatenate((ch3, i[2]), dtype='int32')

    stats['npts'] = len(ch1)
    channels = [ch1, ch2, ch3]

    for i in range(len(channels)):
        stats['channel'] = str(i+1)
        stream += obs.Trace(data=channels[i], header=stats)

    name_mseed = str(stream[0].stats.starttime)[0:4] + str(stream[0].stats.starttime)[5:7]
    name_mseed += str(stream[0].stats.starttime)[8:10] + '_' + str(stream[0].stats.starttime)[11:19].replace(':', '')
    name_mseed += '_' + str(stream[0].stats.station)
    write_name = os.path.join(out_path, name_mseed)
    return stream
    # stream.write(write_name, format='MSEED')


def read_hour_with_gps(in_path, out_path):

    file_names = os.listdir(in_path)
    stream = obs.Stream()

    ch1 = np.array([], dtype='int32')
    ch2 = np.array([], dtype='int32')
    ch3 = np.array([], dtype='int32')

    stats_file = read(in_path, file_names[0])

    for i in stats_file[0].gps_marks:
        if str(i) != 'NonValid':
            time = i.date
            break
    utc_time = obs.UTCDateTime(time[0], time[1], time[2],
                               time[3], time[4], time[5])

    stats = {'station': 'K106'+'-'+str(stats_file.station_id),
             'network': 'T1',
             'location': 'XX',
             'channel': 0,
             'npts': 0,
             'sampling_rate': stats_file.freq,
             'starttime': utc_time,
             'mseed': {'dataquality': 'R'}}

    gps_all = []

    for name in file_names:
        file = read(in_path, name)
        for i in file:
            ch1 = np.concatenate((ch1, i[0]), dtype='int32')
            ch2 = np.concatenate((ch2, i[1]), dtype='int32')
            ch3 = np.concatenate((ch3, i[2]), dtype='int32')

            for j in i.gps_marks:
                if str(j) != 'NonValid':
                    gps_all.append((obs.UTCDateTime(j.date[0], j.date[1], j.date[2],
                                                    j.date[3], j.date[4], j.date[5]), j.height_el))

    ch4 = np.zeros(len(ch1))
    time_axis = [utc_time + i*1/stats['sampling_rate'] for i in range(len(ch1))]
    for i in range(len(gps_all)):
        ch4[time_axis.index(gps_all[i][0])] = gps_all[i][1]

    stats['npts'] = len(ch1)
    channels = [ch1, ch2, ch3, ch4]

    for i in range(len(channels)):
        stats['channel'] = str(i+1)
        stream += obs.Trace(data=channels[i], header=stats)

    name_mseed = str(stream[0].stats.starttime)[0:4] + str(stream[0].stats.starttime)[5:7]
    name_mseed += str(stream[0].stats.starttime)[8:10] + '_' + str(stream[0].stats.starttime)[11:19].replace(':', '')
    name_mseed += '_' + str(stream[0].stats.station)
    write_name = os.path.join(out_path, name_mseed)
    stream.write(write_name, format='MSEED')
