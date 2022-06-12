import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pytictoc import TicToc
Total = TicToc()
Total.tic()

def moving_average(x, w):
    if w%2 == 0:
        w = w - 1
    W = w
    X = np.zeros(x.shape)
    X[0] = x[0]
    X[1] = sum(x[0:3])/3
    idx = 2
    l = 5
    X[2:w - 1] = moving_initial(X, x, W, w, idx, l)
    #X[w-1:] = np.convolve(x, np.ones(w), 'valid') / w
    X[w - 1:] = np.convolve(x, np.ones(w), 'valid') / w # np.ones(w) 와 np.ones((w,)) 는 같은 결과!
    return X

def moving_initial(X, x, W, w, idx, l):
    while w//2 != 1:
        w = w // 2
        X[idx] = sum(x[0:l]) / l
        X[idx + 1] = sum(x[1:l + 1]) / l
        idx += 2
        l += 2
        X[2:W - 1] = moving_initial(X, x, W, w, idx, l)
    return X[2:W - 1]

path_for_accumulation = './output_accumulated'
find_list_accumulated = os.listdir(path_for_accumulation)

ForceAnalysis = False
ForceFinal = False

common_path = './output'
bydate = os.listdir(common_path)
bydate = ['20220120']
print(bydate)
print(len(bydate))
bytime = []
FullBytime = -1 # 하루 안에 실험이 몇 번 이루어졌는지 체크 (한 번이 아닐 수 있어서) <-- Day 또는 Night 안에는 Cartridge 파일이 하나만 있다고 가정.
catridges = []
for d in range(len(bydate)):
    t = TicToc()
    t.tic()

    Isbytime = os.listdir(common_path + '/' + bydate[d])
    LenBytime = 0
    for j in Isbytime:
        if (j == 'Day') or (j == 'Night'):
            bytime.append(j)
            LenBytime += 1
            FullBytime += 1
    print('bytime\n', bytime)
    print('Local len of bytime is ', LenBytime)
    print('Global len of bytime is ', FullBytime)
    for num in range(LenBytime):
        IsCatridge = os.listdir(common_path + '/' + bydate[d] + '/' + bytime[((-1) * LenBytime) + num])
        for k in IsCatridge:
            if os.path.isdir(common_path + '/' + bydate[d] + '/' + bytime[((-1) * LenBytime) + num] + '/' + k):
                catridges.append(k)
        print('catridges\n', catridges)
        data_list = os.listdir(common_path + '/' + bydate[d] + '/' + bytime[((-1) * LenBytime) + num] + '/' + catridges[-1])
        data_csv = [file for file in data_list if (file.endswith('.csv'))]
        print('data_csv\n', data_csv); print(len(data_csv))
        AnalysisPath = common_path + '/' + bydate[d] + '/' + bytime[((-1) * LenBytime) + num] + '/' + catridges[-1]
        print('AnalysisPath\n', AnalysisPath)
        path_split = AnalysisPath.split('/')
        file_list = os.listdir(AnalysisPath)
        csv = [file for file in file_list if (file.endswith('csv'))]

        # Find whether odor & response_by_xvel .csv exist
        find_path = common_path + '/' + bydate[d] + '/' + bytime[((-1) * LenBytime) + num]
        find_list = IsCatridge
        find_csv = [file for file in find_list if (file.endswith('_odor.csv') or file.endswith('_response_by_xvel.csv'))]
        find_final = [file for file in find_list if (file.endswith('_figure_all.png'))]

        find_list_accumulated = os.listdir(path_for_accumulation)

        # Variables setting
        names = catridges[-1].split('_')
        odor_names = names[-5:]
        print(odor_names)
        lane_sort = len(names[1:-6])
        if lane_sort == 1:
            lane_names = [names[1]] * 6
        elif lane_sort == 2:
            lane_names = [names[1]] * 3 + [names[2]] * 3
        elif lane_sort == 3:
            lane_names = [names[1]] * 2 + [names[2]] * 2 + [names[3]] * 2
        elif lane_sort == 6:
            lane_names = names[1:-6]
        print(lane_names)
        rounds = 5  # 실험 한 번에서 냄새당 실행 횟수
        yborders = list(range(100, 1000 + 1, int(900 / len(lane_names))))
        odor = np.zeros((len(csv), len(odor_names) * rounds))
        response_idx = list(range(20, 120 + 1))
        response_by_xvel = np.zeros((len(csv), len(odor_names) * rounds, len(lane_names)))

        if (len(find_csv) == len(csv) + 1) and (ForceAnalysis == False):
            odor = pd.read_csv(find_path + '/' + find_csv[-1], header=None)
            for i in range(len(csv)):
                response_by_xvel[i] = pd.read_csv(find_path + '/' + find_csv[i], header=None)
        else:
            for file_idx in range(len(csv)):  # 0 ~ 6

                path = AnalysisPath + '/' + csv[file_idx]
                data = pd.read_csv(path)
                print('Analyzing ' + path.split('/')[-1])  # (2042605, 7)

                # max_frame_idx = int(data.iloc[-1]['frame']) # 16532
                max_frame_idx = data.at[data.index[-1], 'frame']  # 16532
                # print(max_frame_idx)
                max_fly_idx = max(data['id'])  # 167

                # Collect (fly position + odor signal) at each time step for all flies
                xpos = np.empty((max_fly_idx, max_frame_idx))  # (167, 16532)
                xpos[:] = np.nan
                ypos = np.empty((max_fly_idx, max_frame_idx))
                ypos[:] = np.nan
                ts = np.empty(max_frame_idx)  # (16532, 1)
                ts[:] = np.nan
                valve = np.empty(max_frame_idx)
                valve[:] = np.nan
                last_frame_idx = 0

                for i in range(len(data)):  # 근데 이건 fly index가 일정하게 유지되었다는 가정아래에서 합리적으로 보이는데?????
                    fly_idx = int(data.iloc[i]['id'])
                    frame_idx = int(data.iloc[i]['frame'])
                    xpos[fly_idx - 1][frame_idx - 1] = data.iloc[i]['pos_x']  # (167, 16532)
                    ypos[fly_idx - 1][frame_idx - 1] = data.iloc[i]['pos_y']  # (167, 16532)
                    if (abs(xpos[fly_idx - 1][frame_idx - 1] - xpos[fly_idx - 1][max(0, frame_idx - 2)]) > 25) or (abs(ypos[fly_idx - 1][frame_idx - 1] - ypos[fly_idx - 1][max(0, frame_idx - 2)]) > 25):
                        xpos[fly_idx - 1][frame_idx - 2] = np.nan
                        ypos[fly_idx - 1][frame_idx - 2] = np.nan
                    if last_frame_idx < frame_idx:
                        ts[frame_idx - 1] = data.iloc[i]['timestamp']
                        valve[frame_idx - 1] = data.iloc[i]['valve']
                    last_frame_idx = frame_idx
                # Change valve variable to -1 during light 'ON' (원래는 10000인데) intervals ?????
                valve[valve > 100] = np.nan
                print('valve\n', valve)

                # xpos = pd.DataFrame(np.matrix(xpos))
                # xpos.to_csv('xpos.csv', sep=',', na_rep='NaN', index=False, header=False)

                xpos_diff = np.diff(xpos)  # (167, 16531)
                # xpos_diff = pd.DataFrame(np.matrix(xpos_diff))
                # xpos_diff.to_csv('xpos_diff.csv', sep=',', na_rep='NaN', index=False, header=False)
                ypos_diff = np.diff(ypos)  # (167, 16531)
                lane_num = np.zeros(ypos.shape)  # (167, 16532)
                for i in range(len(lane_names)):
                    lane_num[(ypos < yborders[i + 1]) & (ypos > yborders[i])] = i + 1
                lane_num = lane_num[:, 1:]  # (167, 16531)

                xvel_all = np.nanmean(xpos_diff, axis=0)  # (1, 16531)
                xvel = np.empty((len(lane_names), max_frame_idx - 1))
                xvel[:] = np.nan
                for i in range(len(lane_names)):
                    temp = np.empty(xpos_diff.shape)
                    temp[:] = xpos_diff  # temp = xpos_diff는 안됨!!!
                    # temp = pd.DataFrame(np.matrix(temp))
                    # temp.to_csv('temp' + str(i + 1) + '_test01.csv', sep=',', na_rep='NaN', index=False, header=False)
                    temp[lane_num != (i + 1)] = np.nan
                    # temp.to_csv('temp' + str(i + 1) + '_test02.csv', sep=',', na_rep='NaN', index=False, header=False)
                    xvel[i, :] = np.nanmean(temp, axis=0)

                plt.rcParams['figure.figsize'] = (20, 15)
                plt.rcParams['font.size'] = 12

                f1 = plt.figure()
                plt.subplot(221)
                plt.title(path.split('/')[-1])
                plt.xlabel('time (s)')
                plt.ylabel('x position (pixel)')
                for i in range(max_fly_idx):
                    plt.plot(ts, xpos[i, :])

                ts_shortend = ts[1:]
                plt.subplot(223)
                plt.title('max(valve) = ' + str(int(np.nanmax(valve))))
                plt.xlabel('time (s)')
                plt.ylabel('mean x velocity (pixels/sec)')
                for i in range(len(lane_names)):
                    plt.plot(ts_shortend, 10 * moving_average(xvel[i, :], 9), label=lane_names[i])
                    plt.legend(loc='lower right')
                plt.plot(ts_shortend, 10 * moving_average(xvel_all, 9), label='all')
                plt.legend(loc='lower right')
                plt.plot(ts, 10 * valve, 'k')

                plt.subplot(122)
                plt.title(path.split('/')[-1])
                plt.xlabel('time (s)')
                plt.ylabel('y position (pixel)')
                plt.yticks()
                yline = np.ones(ts.shape)
                for i in range(max_fly_idx):
                    plt.plot(ts, ypos[i, :])
                    for j in range(len(yborders)):
                        plt.plot(ts, yline * yborders[j], 'k')

                number_xbins = 10
                number_ybins = len(lane_names)  # 6
                ybins = np.zeros((len(lane_names), 1, (yborders[1] - yborders[0]) + 1))
                for i in range(len(lane_names)):
                    ybins[i][:] = list(range(yborders[i], yborders[i + 1] + 1))
                xmin = np.nanmin(xpos)  # np.nanmin(xpos[:]) 와 같은 듯!
                xmax = np.nanmax(xpos)
                xborders = np.ceil(np.linspace(xmin, xmax, number_xbins + 1))
                fly_counts = np.zeros((len(lane_names), number_xbins, max_frame_idx))
                for frame_idx in range(max_frame_idx):
                    xpos_here = np.empty(xpos[:, frame_idx].shape)
                    xpos_here[:] = xpos[:, frame_idx]
                    xpos_here = xpos_here[~np.isnan(xpos_here)]
                    ypos_here = np.empty(ypos[:, frame_idx].shape)
                    ypos_here[:] = ypos[:, frame_idx]
                    ypos_here = ypos_here[~np.isnan(ypos_here)]
                    for i in range(len(lane_names)):
                        for j in range(number_xbins):
                            fly_counts[i][j, frame_idx] = (
                                        (xpos_here > xborders[j]) & (xpos_here <= xborders[j + 1]) & (
                                            ypos_here >= np.min(ybins[i])) & (ypos_here <= np.max(ybins[i]))).sum()

                f2 = plt.figure()
                f2.suptitle(path.split('/')[-1])
                for i in range(len(lane_names)):
                    plt.subplot(3, int(len(lane_names) / 3), i + 1)
                    plt.title('lane#' + str(i + 1) + ' - ' + lane_names[i])
                    for j in range(number_xbins):
                        if j < 2:
                            plt.plot(ts, fly_counts[i][j, :], label='xbin#' + str(j + 1), color='r')
                            plt.legend(loc='upper right')
                        else:
                            plt.plot(ts, fly_counts[i][j, :], label='xbin#' + str(j + 1), alpha=.5)
                            plt.legend(loc='upper right')
                    plt.plot(ts, valve, 'k')

                f1.savefig(AnalysisPath + '/' + path.split('/')[-1][:-4] + '_figure1.png', bbox_inches='tight', dpi=300)
                f2.savefig(AnalysisPath + '/' + path.split('/')[-1][:-4] + '_figure2.png', bbox_inches='tight', dpi=300)

                puff_start_idx = np.where(np.concatenate(([0], np.diff(valve)), axis=None) > 0)[0]
                print('puff_start_idx\n', puff_start_idx)
                for i in range(len(puff_start_idx)):  # 25
                    odor[file_idx, i] = valve[puff_start_idx[i]]  # (7,25)
                    for j in range(len(lane_names)):  # 6
                        response_by_xvel[file_idx][i, j] = np.nanmean(xvel[j, puff_start_idx[i] + response_idx])  # (7,25,6)
                odor = odor.astype(int)
                print('odor\n', odor)
                print(type(odor))

            odor = pd.DataFrame(odor)
            AccumulatedPath1 = path_for_accumulation + '/' + bydate[d] + '_' + lane_names[0] + '_' + bytime[((-1) * LenBytime) + num] + '_' +\
                               odor_names[0] + '_' + odor_names[1] + '_' + odor_names[2] + '_' + odor_names[3] + '_' + odor_names[4]
            AccumulatedPath2 = path_for_accumulation + '/' + bydate[d] + '_' + lane_names[0] + '_' + lane_names[3] + '_' + bytime[((-1) * LenBytime) + num] + '_' +\
                               odor_names[0] + '_' + odor_names[1] + '_' + odor_names[2] + '_' + odor_names[3] + '_' + odor_names[4]
            AccumulatedPath3 = path_for_accumulation + '/' + bydate[d] + '_' + lane_names[0] + '_' + lane_names[2] + '_' + lane_names[4] + '_' + bytime[((-1) * LenBytime) + num] + '_' +\
                               odor_names[0] + '_' + odor_names[1] + '_' + odor_names[2] + '_' + odor_names[3] + '_' + odor_names[4]
            AccumulatedPath6 = path_for_accumulation + '/' + bydate[d] + '_' + lane_names[0] + '_' + lane_names[1] + '_' + lane_names[2] + '_' + lane_names[3] + '_' + lane_names[4] + '_' + lane_names[5] +\
                               '_' + bytime[((-1) * LenBytime) + num] + '_' + odor_names[0] + '_' + odor_names[1] + '_' + odor_names[2] + '_' + odor_names[3] + '_' + odor_names[4]
            if lane_sort == 1:
                AccumulatedPath = AccumulatedPath1
            elif lane_sort == 2:
                AccumulatedPath = AccumulatedPath2
            elif lane_sort == 3:
                AccumulatedPath = AccumulatedPath3
            elif lane_sort == 6:
                AccumulatedPath = AccumulatedPath6
            odor.to_csv(AnalysisPath.split('_')[0] + '_odor.csv', sep=',', na_rep='NaN', index=False, header=False)
            odor.to_csv(AccumulatedPath + '_odor.csv', sep=',', na_rep='NaN', index=False, header=False)
            for count in range(len(csv)):
                response_by_xvel_save = pd.DataFrame(response_by_xvel[count])
                response_by_xvel_save.to_csv(AnalysisPath.split('_')[0] + '_' + str(count) + '_response_by_xvel.csv', sep=',', na_rep='NaN', index=False, header=False)
                response_by_xvel_save.to_csv(AccumulatedPath + '_' + str(count) + '_response_by_xvel.csv', sep=',', na_rep='NaN', index=False, header=False)

        if (len(find_final) == 0) or (ForceFinal == True):
            odor = np.array(odor)

            plt.rcParams['figure.figsize'] = (20, 15)
            plt.rcParams['font.size'] = 12
            width = 0.2

            f3 = plt.figure()
            f3.suptitle(path_split[2] + '-' + path_split[3])
            response_mean = np.zeros((len(lane_names), len(odor_names), len(csv)))  # (6,5,7)
            idx = np.zeros(rounds)
            RM_3d = np.zeros((len(lane_names), len(odor_names), len(csv)))
            for i in range(len(lane_names)):  # 6
                response_all = np.zeros((len(odor_names), len(csv), 1, rounds))  # (5,7,5,1)
                for j in range(len(odor_names)):  # 5
                    RA = np.zeros((len(csv), rounds))
                    for k in range(len(csv)):  # 7
                        idx[:] = np.where(odor[k, :] == (j + 1))[0]
                        idx = idx.astype(int)
                        response_all[j, k] = (-1) * response_by_xvel[k][idx, i]
                        response_mean[i][j, k] = np.mean(response_all[j, k])
                        RA[k] = pd.DataFrame(response_all[j, k])
                    RA = pd.DataFrame(RA)
                    plt.figure(f3)
                    plt.subplot(3, int(len(lane_names) / 3), i + 1)
                    for b in range(rounds):
                        plt.scatter(np.linspace(((2 * j) + 1), ((2 * j) + 1 + 0.2 * (len(csv) - 1)), len(csv)),
                                    10 * RA[b], marker='.', c='k')
                RM = pd.DataFrame(response_mean[i])
                RM_3d[i] = 10 * response_mean[i]
                print('RM\n', RM)
                plt.figure(f3)
                plt.subplot(3, int(len(lane_names) / 3), i + 1)
                plt.title('lane#' + str(i + 1) + ' - ' + lane_names[i])
                plt.ylabel('Mean velocity (pixels/sec)')
                plt.ylim([-20, 50])
                xticks = np.linspace(1, 2 * len(odor_names) - 1, len(odor_names))
                for a in range(len(csv)):
                    plt.bar(xticks + width * a * np.ones(len(odor_names)), 10 * RM[a], width=width, alpha=0.7)
                zero_line = np.linspace(0, 2 * len(odor_names) + 1, len(odor_names))
                plt.plot(zero_line, np.zeros(zero_line.shape), c='k')
                plt.xticks(xticks + (width / 2) * (len(csv) - 1) * np.ones(len(odor_names)), odor_names)
            f4 = plt.figure()
            ax = plt.axes(projection='3d')
            ax.set_xlabel(lane_names[0])
            ax.set_ylabel(lane_names[2])
            ax.set_zlabel(lane_names[4])
            ax.set_xlim([-10, 30])
            ax.set_ylim([-10, 30])
            ax.set_zlim([-10, 30])
            for i in range(len(odor_names)):
                if i == 0:
                    ax.scatter3D(RM_3d[0][i], RM_3d[2][i], RM_3d[4][i], c='b', label=odor_names[i])
                    ax.scatter3D(RM_3d[1][i], RM_3d[3][i], RM_3d[5][i], c='b')
                elif i == 1:
                    ax.scatter3D(RM_3d[0][i], RM_3d[2][i], RM_3d[4][i], c='r', label=odor_names[i])
                    ax.scatter3D(RM_3d[1][i], RM_3d[3][i], RM_3d[5][i], c='r')
                elif i == 2:
                    ax.scatter3D(RM_3d[0][i], RM_3d[2][i], RM_3d[4][i], c='c', label=odor_names[i])
                    ax.scatter3D(RM_3d[1][i], RM_3d[3][i], RM_3d[5][i], c='c')
                elif i == 3:
                    ax.scatter3D(RM_3d[0][i], RM_3d[2][i], RM_3d[4][i], c='m', label=odor_names[i])
                    ax.scatter3D(RM_3d[1][i], RM_3d[3][i], RM_3d[5][i], c='m')
                elif i == 4:
                    ax.scatter3D(RM_3d[0][i], RM_3d[2][i], RM_3d[4][i], c='g', label=odor_names[i])
                    ax.scatter3D(RM_3d[1][i], RM_3d[3][i], RM_3d[5][i], c='g')
                ax.legend(loc='best')

            f3.savefig(AnalysisPath.split('_')[0] + '_figure_all.png', bbox_inches='tight', dpi=300)
            f4.savefig(AnalysisPath.split('_')[0] + '_3Dfigure_all.png', bbox_inches='tight', dpi=300)
    plt.close(); plt.close()
    t.toc('One day file took'); print('')

Total.toc('All Process took')

#plt.show()