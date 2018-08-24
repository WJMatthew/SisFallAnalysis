import pandas as pd
import numpy as np
import warnings
import time
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

path = '/Users/mattjohnson/Desktop/Python2018/sisfall/SubjectDataFrames/acm_SA'

subjectList = []

firstIndex = 1
lastIndex = 4

print('*',firstIndex, 'to', lastIndex-1, '...')

for i in range(firstIndex, lastIndex):
    data = pd.read_csv(path + str(i).zfill(2) + '.csv')
    data = data.drop('Unnamed: 0', axis=1)
    df = data[['x1', 'y1', 'z1', 'x2', 'y2', 'z2', 'activity', 'subject', 'trial']]
    subjectList.append(df)

# Codes for ADLs
dailies = ['D01', 'D02', 'D03', 'D04', 'D05', 'D06', 'D07', 'D08', 'D09', 'D10', 'D11', 'D12', 'D13', 'D14', 'D15',
           'D16', 'D17', 'D18', 'D19']
# Codes for Falls
falls = ['F01', 'F02', 'F03', 'F04', 'F05', 'F06', 'F07', 'F08', 'F09', 'F10', 'F11', 'F12', 'F13', 'F14', 'F15']

# Lists to hold dataframes sorted by activity (ADLs and Falls)
adl_list = []
fall_list = []
# Iterate through subject data and sort into ADLs and Falls
for s in subjectList:
    for d in dailies:
        tempdf = s[s['activity'] == d]
        adl_list.append(tempdf)

    for f in falls:
        tempdf = s[s['activity'] == f]
        fall_list.append(tempdf)


# Titles for fall activities
fall_titles = ['Fall forward while walking caused by a slip', 'Fall backward while walking caused by a slip',
    'Lateral fall while walking caused by a slip', 'Fall forward while walking caused by a trip',
    'Fall forward while jogging caused by a trip', 'Vertical fall while walking caused by fainting',
    'Fall while walking, with use of hands in a table to dampen fall, caused by fainting',
    'Fall forward when trying to get up', 'Lateral fall when trying to get up',
    'Fall forward when trying to sit down', 'Fall backward when trying to sit down', 'Lateral fall when trying to sit down',
    'Fall forward while sitting, caused by fainting or falling asleep',
    'Fall backward while sitting, caused by fainting or falling asleep',
    'Lateral fall while sitting, caused by fainting or falling asleep']
# Titles for ADLs
adl_titles = ['Walking slowly', 'Walking quickly', 'Jogging slowly', 'Jogging quickly', 'Walking upstairs and downstairs slowly',
    'Walking upstairs and downstairs quickly','Slowly sit in a half height chair, wait a moment, and up slowly',
    'Quickly sit in a half height chair, wait a moment, and up quickly',
    'Slowly sit in a low height chair, wait a moment, and up slowly','Quickly sit in a low height chair, wait a moment, and up quickly',
    'Sitting a moment, trying to get up, and collapse into a chair',
    'Sitting a moment, lying slowly, wait a moment, and sit again','Sitting a moment, lying quickly, wait a moment, and sit again',
    'Being on oneís back change to lateral position, wait a moment, and change to oneís back',
    'Standing, slowly bending at knees, and getting up', 'Standing, slowly bending without bending knees, and getting up',
    'Standing, get into a car, remain seated and get out of the car','Stumble while walking',
    'Gently jump without falling (trying to reach a high object)']


from scipy.signal import butter, lfilter, freqz

# Filter requirements.
order = 4
fs = 200.0  # sample rate, Hz
cutoff = 5.0  # desired cutoff frequency of the filter, Hz


# From??????
def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y


# Codes for trials
trials = ['R01', 'R02', 'R03', 'R04', 'R05']
horiz_std_mag_THRESHOLD = 170  # Set threshold for Horizontal Standard Deviation Magnitude
horiz_vec_mag_THRESHOLD = 400  # Set threshold for Horizontal Sum Vector Magnitude
vector_mag_THRESHOLD = 900  # Set threshold for Sum Vector Magnitude
gyro_horiz_std_THRESHOLD = 170

# Lists to hold prepared fall and ADL dataframes
fall_df_list = []
adl_df_list = []
# Lists to hold sliding windows of 1.25s with a 50% overlap
windowList_adl = []
windowList_fall = []

windowHighlights_adl = []
falseAlarms = []


# Method that takes in a dataframe and kind=['f', 'a'] and stores sliding windows of 1.25s
# with a 50% overlap
def sliding_window(dataframe, kind):
    k = 0
    for i in range(0, len(dataframe) - 256, 128):
        w = 256  # Size of sliding window (256 points at 200Hz = 1.25 seconds)
        w1 = dataframe.iloc[i:i + w][:]

        if ((w1['horiz_std_mag9'].max() >= horiz_std_mag_THRESHOLD) * 1 + (
                w1['horiz_vector_mag9'].max() >= horiz_vec_mag_THRESHOLD) * 1
                + (w1['vm'].max() >= vector_mag_THRESHOLD) * 1 + (
                        w1['gyro_horiz_std_mag'].max() >= gyro_horiz_std_THRESHOLD) * 1 >= 3):
            w1['Fall'] = 1

            if kind == 'a':
                falseAlarms.append(k)
                print('k:', k, 'i:', i, '\n')
                print('False Alarm: hstd9:', w1['horiz_std_mag9'].max(), 'hvm9:', w1['horiz_vector_mag9'].max())
                print('\tvm:', w1['vm'].max(), 'hvm9:', 'ghstd:', w1['gyro_horiz_std_mag'].max())
                windowHighlights_adl.append(1)
        else:
            w1['Fall'] = 0
            if kind == 'a':
                windowHighlights_adl.append(0)

        if kind == 'f':  # If fall
            windowList_fall.append(w1)
        else:  # If ADL
            windowList_adl.append(w1)
        k += 1


# Method that takes in a kind=['f', 'a'] and prepares respective data
def prepare_data(kind):
    if kind == 'f':  # If fall
        putList, takeList = fall_df_list, fall_list
    else:  # If ADL
        putList, takeList = adl_df_list, adl_list

    start = time.time()  # Timer for testing

    for i in range(0, len(takeList)):  # Iterate through the takeList (fall_list or adl_list)

        my_df = takeList[i].copy()  # copy dataframe from list
        new_df = pd.DataFrame()  # placeholder

        for trial in trials:  # Iterate through trials (1-5)

            # Get relevant trial data
            trial_df = my_df[my_df['trial'] == trial]

            tempdf = pd.DataFrame()  # dataframe for putting into filter
            # Low Pass Buttersworth Filter and remove bias
            tempdf['ax'], tempdf['ay'], tempdf['az'] = trial_df['x1'], trial_df['y1'], trial_df['z1']
            tempdf = tempdf.reset_index(drop=True)
            tempdf['fx'] = pd.Series(butter_lowpass_filter(trial_df['x1'], cutoff, fs, order))
            tempdf['fy'] = pd.Series(butter_lowpass_filter(trial_df['y1'], cutoff, fs, order))
            tempdf['fz'] = pd.Series(butter_lowpass_filter(trial_df['z1'], cutoff, fs, order))
            tempdf['bx'] = tempdf['fx'].diff()
            tempdf['by'] = tempdf['fy'].diff()
            tempdf['bz'] = tempdf['fz'].diff()

            tempdf = tempdf.reset_index(drop=True)
            trial_df = trial_df.reset_index(drop=True)
            tempdf['gx'], tempdf['gy'], tempdf['gz'] = trial_df['x2'], trial_df['y2'], trial_df['z2']

            # Rolling averages
            tempdf['y_roll'] = pd.Series(tempdf['by'].rolling(200).mean())
            tempdf['fy_roll'] = pd.Series(tempdf['fy'].rolling(200).mean())
            tempdf['gy_roll'] = pd.Series(tempdf['by'].rolling(200).mean())

            # Rolling standard deviations
            tempdf['bx_std'] = tempdf['bx'].rolling(200).std()
            tempdf['by_std'] = tempdf['by'].rolling(200).std()
            tempdf['bz_std'] = tempdf['bz'].rolling(200).std()
            tempdf['fx_std'] = tempdf['fx'].rolling(200).std()
            tempdf['fy_std'] = tempdf['fy'].rolling(200).std()
            tempdf['fz_std'] = tempdf['fz'].rolling(200).std()
            tempdf['gx_std'] = tempdf['fx'].rolling(200).std()
            tempdf['gy_std'] = tempdf['fy'].rolling(200).std()
            tempdf['gz_std'] = tempdf['fz'].rolling(200).std()


            # Integral stuff
            tempdf['xsum'] = pd.expanding_sum(((abs(tempdf['ax']).rolling(2).sum() / 2) * (1 / 200)).fillna(0))
            tempdf['ysum'] = pd.expanding_sum(((abs(tempdf['ay']).rolling(2).sum() / 2) * (1 / 200)).fillna(0))
            tempdf['zsum'] = pd.expanding_sum(((abs(tempdf['az']).rolling(2).sum() / 2) * (1 / 200)).fillna(0))
            tempdf['time'] = 1 / 200
            tempdf['time'] = pd.expanding_sum(tempdf['time'])
            # C10 Signal Magnitude Area
            tempdf['SigMagArea'] = (tempdf['xsum'] + tempdf['ysum'] + tempdf['zsum']) / tempdf['time']
            # C11
            tempdf['HorizSigMagArea'] = (tempdf['xsum'] + tempdf['zsum']) / tempdf['time']
            # Sum vector magnitude
            tempdf['vm'] = np.sqrt(tempdf['fx'] ** 2 + tempdf['fy'] ** 2 + tempdf['fz'] ** 2)
            # Maximum peak to peak acceleration amplitude
            tempdf['Amax'] = (tempdf['vm'].rolling(200).max())
            tempdf['Amin'] = (tempdf['vm'].rolling(200).min())
            # C3
            tempdf['peak_diff'] = tempdf['Amax'] - tempdf['Amin']
            # Angle from horizontal to z-axis
            tempdf['angle_from_horiz'] = np.arctan2(np.sqrt(tempdf['fx'] ** 2 + tempdf['fz'] ** 2), -tempdf['fy']) * 180 / np.pi
            tempdf['angle_std'] = pd.rolling_std(tempdf['angle_from_horiz'], 200)

            # had to make versions of this to put into sliding window, will change once I
            # confirm they're the same as the others below
            tempdf['horiz_std_mag9'] = np.sqrt(tempdf['fx_std'] ** 2 + tempdf['fz_std'] ** 2)
            tempdf['horiz_vector_mag9'] = np.sqrt(tempdf['fx'] ** 2 + tempdf['fz'] ** 2)
            tempdf['std_mag9'] = np.sqrt(tempdf['fx_std'] ** 2 + tempdf['fy_std'] ** 2 + tempdf['fz_std'] ** 2)
            tempdf['diff_std_mag9'] = np.sqrt(tempdf['bx_std'] ** 2 + tempdf['by_std'] ** 2 + tempdf['bz_std'] ** 2)
            tempdf['horiz_mag2'] = np.sqrt(tempdf['bx'] ** 2 + tempdf['bz'] ** 2)
            tempdf['horiz_std_mag2'] = np.sqrt(tempdf['bx_std'] ** 2 + tempdf['bz_std'] ** 2)
            tempdf['vector_mag2'] = np.sqrt(tempdf['bx'] ** 2 + tempdf['by'] ** 2 + tempdf['bz'] ** 2)

            tempdf['gyro_horiz_std_mag'] = np.sqrt(tempdf['gx_std'] ** 2 + tempdf['gz_std'] ** 2)
            tempdf['gyro_vector_mag'] = np.sqrt(tempdf['gx'] ** 2 + tempdf['gy'] ** 2 + tempdf['gz'] ** 2)
            tempdf['gyro_horiz_mag'] = np.sqrt(tempdf['gx'] ** 2 + tempdf['gz'] ** 2)
            tempdf['gyro_std_mag'] = np.sqrt(tempdf['gx_std'] ** 2 + tempdf['gy_std'] ** 2 + tempdf['gz_std'] ** 2)

            tempdf = pd.concat(
                [tempdf.reset_index(drop=True), trial_df[['activity', 'subject', 'trial']].reset_index(drop=True)],
                axis=1)
            new_df = pd.concat([new_df.reset_index(drop=True), tempdf])

            sliding_window(tempdf, kind)

        # differential vector mag
        new_df['vector_mag'] = np.sqrt(new_df['fx'] ** 2 + new_df['fy'] ** 2 + new_df['fz'] ** 2)
        # C2
        new_df['horiz_mag'] = np.sqrt(new_df['fx'] ** 2 + new_df['fz'] ** 2)
        #
        new_df['vert'] = new_df['by'] - new_df['y_roll']
        new_df['vert2'] = new_df['ay'] - new_df['y_roll']
        new_df['vert3'] = new_df['fy'] - new_df['fy_roll']
        # C9
        new_df['std_mag2'] = np.sqrt(new_df['bx_std'] ** 2 + new_df['by_std'] ** 2 + new_df['bz_std'] ** 2)

        putList.append(new_df.fillna(0))

    print('Completed... It took', time.time() - start, 'seconds.')


prepare_data('f') # Prepare Fall Data
prepare_data('a') # Prepare ADL Data

print('ADL dataframes:',len(adl_df_list), '\t\tFall dataframes:', len(fall_df_list))
print('ADL windows:', len(windowList_adl), '\t\tFall windows:', len(windowList_fall))

wList_f = windowList_fall[:]     # Copy of fall window list
wList_a = windowList_adl[:]

# putting together falls and adls
fall_df = pd.concat(fall_df_list)
adl_df = pd.concat(adl_df_list)
all_df = pd.concat([fall_df, adl_df]).fillna(0) # Filling nulls with zeroes

horiz_std_mag_THRESHOLD = 155  # Set threshold for Horizontal Standard Deviation Magnitude
horiz_vec_mag_THRESHOLD = 400  # Set threshold for Horizontal Sum Vector Magnitude
vector_mag_THRESHOLD = 750  # Set threshold for Sum Vector Magnitude
gyro_horiz_std_THRESHOLD = 150
belowThresh = 0  # Keep track of windows below thresholds
aboveThresh = 0  # Keep track of windows above thresholds
i = 0  # placeholder
lastActInd = 0  # Last activity index
lastTrialNum = 0  # Last trial number
fallTrialList = []  #
listInFallTrialList = []  #
missedFalls = 0

for window in wList_f:  # Iterate through fall windows
    windNum = i  # Set window index
    respWindNum = i % 22  # Set respective window index (0-21)
    activityIndex = int(i / (22 * 5))  # Calculate activity index (0-13/14?)
    trialNum = (int(i / 22)) % 5  # Calculate trial number (0-4)

    # Custom setting for thresholds, currently need to pass 2/3 thresholds to pass
    if ((window['horiz_std_mag9'].max() >= horiz_std_mag_THRESHOLD) * 1 + (
            window['horiz_vector_mag9'].max() >= horiz_vec_mag_THRESHOLD) * 1
            + (window['vm'].max() >= vector_mag_THRESHOLD) * 1 + (
                    window['gyro_horiz_std_mag'].max() >= gyro_horiz_std_THRESHOLD) * 1 >= 1):

        aboveThresh += 1

        if (activityIndex == lastActInd):
            if (trialNum == lastTrialNum):
                listInFallTrialList.append(respWindNum)
                lastTrialNum = trialNum
                lastActInd = activityIndex
            else:
                fallTrialList.append(listInFallTrialList)
                listInFallTrialList = []
                listInFallTrialList.append(respWindNum)
                lastActInd = activityIndex
                diff = (trialNum - lastTrialNum)
                absdiff = diff % 5

                if absdiff > 1:
                    for j in range(absdiff - 1):
                        fallTrialList.append([])
                        missedFalls += 1

                lastTrialNum = trialNum
        else:
            diff = (trialNum - lastTrialNum)
            absdiff = diff % 5
            activity_diff = (activityIndex - lastActInd)

            fallTrialList.append(listInFallTrialList)
            if (activity_diff > 0) & (trialNum != 0):
                for k in range(3 - absdiff):
                    fallTrialList.append([])
                    missedFalls += 1
            if absdiff > 1:
                for j in range(absdiff - 1):
                    fallTrialList.append([])
                    missedFalls += 1
            listInFallTrialList = []
            listInFallTrialList.append(respWindNum)
            lastTrialNum = trialNum
            lastActInd = activityIndex
    else:
        belowThresh += 1
    i += 1

fallTrialList.append(listInFallTrialList)

print('below:', belowThresh)
print('above:', aboveThresh)


FTL = fallTrialList[:]
print('len fallTrialList:', len(FTL))
print('missed falls:', missedFalls)
print('false alarms:', len(falseAlarms), ':', falseAlarms)


fall_windows = pd.concat(wList_f)
adl_windows = pd.concat(wList_a)
all_windows = pd.concat([fall_windows, adl_windows])


print('COMPLETED')



# Titles for fall activities
fall_titles = ['Fall forward while walking caused by a slip', 'Fall backward while walking caused by a slip',
    'Lateral fall while walking caused by a slip', 'Fall forward while walking caused by a trip',
    'Fall forward while jogging caused by a trip', 'Vertical fall while walking caused by fainting',
    'Fall while walking, with use of hands in a table to dampen fall, caused by fainting',
    'Fall forward when trying to get up', 'Lateral fall when trying to get up',
    'Fall forward when trying to sit down', 'Fall backward when trying to sit down', 'Lateral fall when trying to sit down',
    'Fall forward while sitting, caused by fainting or falling asleep',
    'Fall backward while sitting, caused by fainting or falling asleep',
    'Lateral fall while sitting, caused by fainting or falling asleep']
# Titles for ADLs
adl_titles = ['Walking slowly', 'Walking quickly', 'Jogging slowly', 'Jogging quickly', 'Walking upstairs and downstairs slowly',
    'Walking upstairs and downstairs quickly','Slowly sit in a half height chair, wait a moment, and up slowly',
    'Quickly sit in a half height chair, wait a moment, and up quickly',
    'Slowly sit in a low height chair, wait a moment, and up slowly','Quickly sit in a low height chair, wait a moment, and up quickly',
    'Sitting a moment, trying to get up, and collapse into a chair',
    'Sitting a moment, lying slowly, wait a moment, and sit again','Sitting a moment, lying quickly, wait a moment, and sit again',
    'Being on oneís back change to lateral position, wait a moment, and change to oneís back',
    'Standing, slowly bending at knees, and getting up', 'Standing, slowly bending without bending knees, and getting up',
    'Standing, get into a car, remain seated and get out of the car','Stumble while walking',
    'Gently jump without falling (trying to reach a high object)']

dailies = ['D01', 'D02', 'D03', 'D04', 'D05', 'D06', 'D07', 'D08', 'D09', 'D10', 'D11', 'D12', 'D13', 'D14', 'D15',
           'D16', 'D17', 'D18', 'D19']
falls = ['F01', 'F02', 'F03', 'F04', 'F05', 'F06', 'F07', 'F08', 'F09', 'F10', 'F11', 'F12', 'F13', 'F14', 'F15']

fs = 200.0  # frequency samplet
subjectCodes = ['SA01', 'SA02', 'SA03', 'SA04']


def get_trial_time(index, kind):
    r = 5
    if kind == 'f':
        n = 3000.0
        t = np.linspace(0, 15.0, n, endpoint=False)
    else:
        if index in list(range(0, 4)):
            T = 100  # seconds
        elif index in [4, 5, 16]:
            T = 25
        else:
            T = 12

        n = int(T * fs)  # total number of samples
        t = np.linspace(0, T, n, endpoint=False)

        if index <= 3:
            r = 1

    return t, int(n), int(r)


## index w <-- activity = F[w+1], eg. index=4 gives a df that contains activity F05
## This makes sense since data_list[0] contains activity F01

def plot_trials(index, subjectIndex, kind):
    if kind == 'f':
        correctList, correctTitles, correctCodes = fall_df_list, fall_titles, falls
    else:
        correctList, correctTitles, correctCodes = adl_df_list, adl_titles, dailies

    new_df = correctList[index + 15 * subjectIndex]
    t, n, r = get_trial_time(index, kind)

    T = int(n / fs)
    l = list(range(0, int(1000 * T), int(1000 * 0.625)))
    l = np.array(l) / 1000
    xcoords = list(l)

    plt.figure(figsize=(15, 2 * r))

    for i in range(0, r):
        curr_df = new_df[i * n:i * n + n]
        if len(curr_df) != 0:
            if len(curr_df) != n:
                curr_df = curr_df.append(curr_df.iloc[len(curr_df) - 1])
            plt.subplot(r, 1, i + 1)
            plt.plot(t, curr_df['fx'], 'b-', label='x')
            plt.plot(t, curr_df['fy'], 'r-', label='y')
            plt.plot(t, curr_df['fz'], 'y-', label='z')
            for xc in xcoords:
                plt.axvline(x=xc)
            plt.grid()
            labs = list(range(0, len(xcoords) - 2))
            plt.xticks(xcoords, labs)
            plt.legend()
            plt.ylabel('trial' + str(i + 1))
            if i == 0:
                plt.title(subjectCodes[subjectIndex] + ' - ' + correctCodes[index] + ':' + correctTitles[index])

        if kind == 'f':
            tempA = FTL[5 * index + i + 75 * subjectIndex]
            if len(tempA) > 0:
                shadeStart = tempA[0]
                shadeFin = tempA[len(tempA) - 1] + 2
                plt.axvspan(shadeStart * .625, shadeFin * .625, color='red', alpha=0.5)

    plt.subplots_adjust(hspace=0.35)
    plt.show()


def plot_one_from_each(index, kind):
    if kind == 'f':
        correctList, correctTitles, correctCodes = fall_df_list, fall_titles, falls
    else:
        correctList, correctTitles, correctCodes = adl_df_list, adl_titles, dailies

    plt.figure(figsize=(15, 10))

    for i in range(0, 5):
        if (index + i + 1) == len(correctList): return
        new_df = correctList[index + i]
        t, n, r = get_trial_time(index, kind)

        try:
            curr_df = new_df[0:n]
            if len(curr_df) != n:
                curr_df = curr_df.append(curr_df.iloc[len(curr_df) - 1])
            plt.subplot(5, 1, i + 1)
            plt.plot(t, curr_df['bx'], 'b-', label='x')
            plt.plot(t, curr_df['by'], 'r-', label='y')
            plt.plot(t, curr_df['bz'], 'y-', label='z')
            plt.grid()
            plt.legend()
            plt.ylabel('Acc')
            plt.title(correctCodes[index + i] + ' ' + correctTitles[index + i])
        except:
            print('')

    plt.subplots_adjust(hspace=0.4)
    plt.show()


def plot_feats(index, tri, kind):
    if kind == 'f':
        correctList, correctTitles, correctCodes = fall_df_list, fall_titles, falls
    else:
        correctList, correctTitles, correctCodes = adl_df_list, adl_titles, dailies

    t, n, r = get_trial_time(index, kind)

    new_df = correctList[index]
    curr_df = new_df[tri * n:tri * n + n]

    feat_list = ['vector_mag', 'vector_mag2', 'horiz_mag', 'vert', 'std_mag9', 'horiz_std_mag9',
                 'peak_diff', 'HorizSigMagArea', 'angle_from_horiz', 'gyro_horiz_std_mag']
    colour_list = ['b-', 'r-', 'k-', 'c-', 'C2', 'C4', 'C1', 'C5', 'C6', 'C7']

    x = len(feat_list)
    plt.figure(figsize=(15, 2 * x))

    for i, feat, colour in zip(range(0, x), feat_list, colour_list):
        plt.subplot(x, 1, i + 1)
        plt.plot(t, curr_df[feat], colour, label=feat)
        plt.grid()
        plt.legend()
        plt.ylabel(feat)
        if i == 0: plt.title(correctCodes[index] + ' ' + correctTitles[index])

    plt.subplots_adjust(hspace=0.35)
    plt.show()


def plot_trial(index, tri, kind):
    if kind == 'f':
        correctList, correctTitles, correctCodes = fall_df_list, fall_titles, falls
    else:
        correctList, correctTitles, correctCodes = adl_df_list, adl_titles, dailies

    new_df = correctList[index]

    t, n, r = get_trial_time(index, kind)
    plt.figure(figsize=(15, 24))

    xtypes = ['ax', 'fx', 'bx', 'gx']
    ytypes = ['ay', 'fy', 'by', 'gy']
    ztypes = ['az', 'fz', 'bz', 'gz']
    ylabs = ['raw acc (m/s^2)', 'filtered', 'filt differential', 'gyro']
    for i in range(0, len(xtypes)):
        curr_df = new_df[tri * n:tri * n + n]
        if len(curr_df) != n:
            curr_df = curr_df.append(curr_df.iloc[len(curr_df) - 1])
        plt.subplot(12, 1, i + 1)
        plt.plot(t, curr_df[xtypes[i]], 'b-', label='x')
        plt.plot(t, curr_df[ytypes[i]], 'r-', label='y')
        plt.plot(t, curr_df[ztypes[i]], 'y-', label='z')
        plt.grid()
        plt.legend()
        plt.ylabel(ylabs[i])
        if i == 0: plt.title(correctCodes[index] + ' ' + correctTitles[index])

    curr_df = correctList[index][0:int(n)]
    if len(curr_df) != n:
        curr_df = curr_df.append(curr_df.iloc[len(curr_df) - 1])

    feat_list = ['vector_mag', 'vector_mag2', 'horiz_mag', 'vert', 'std_mag9', 'horiz_std_mag9',
                 'peak_diff', 'HorizSigMagArea', 'angle_from_horiz', 'gyro_horiz_std_mag']
    colour_list = ['b-', 'r-', 'k-', 'c-', 'C2', 'C4', 'C1', 'C5', 'C6', 'C7']

    x = len(feat_list) + len(xtypes)
    plt.figure(figsize=(15, 2 * x))

    for i, feat, colour in zip(range(0, x), feat_list, colour_list):
        plt.subplot(12, 1, i + 1)
        plt.plot(t, curr_df[feat], colour, label=feat)
        plt.grid()
        plt.legend()
        plt.ylabel(feat)
        if i == 0: plt.title(correctCodes[index] + ' ' + correctTitles[index])

    plt.subplots_adjust(hspace=0.35)
    plt.show()

