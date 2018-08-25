*[In progress]*

# Analysis of the SisFall Dataset for Fall Detection
**Matthew Johnson, July 25, 2018 (last updated August 24, 2018)**

#### A dataset of performed trials of activities of daily living (ADLs) and falls with subjects wearing two triaxis accelerometers and a gyroscope.
<br>
I used features common throughout related literature, specifically from [3]:<br><br>

**Sum Vector Magnitude** =  ![Image](https://i.imgur.com/cte8bDa.gif)
<br><br>

**Sum Vector Magnitude on Horizontal Plane** = ![Image](https://i.imgur.com/HAuhKDf.gif)
<br><br>

**Angle between z-axis and vertical** = ![Image](https://i.imgur.com/QuFoizh.gif)
<br><br>

**Standard deviation magnitude on horizontal plane** = ![Image](https://i.imgur.com/Zp3A9aG.gif)
<br><br>

**Standard deviation magnitude** = ![Image](https://i.imgur.com/edDMmHI.gif)
<br><br>

**Signal Magnitude Area** = ![Image](https://i.imgur.com/EU9NIyQ.gif)
<br><br>

**Signal Magnitude Area on horizontal plane** = ![Image](https://i.imgur.com/3wFkSlu.gif)
<br><br>

SisFall Dataset: http://sistemic.udea.edu.co/en/investigacion/proyectos/english-falls/<br>
##### Sources: <br>
* [1] **Automatic Fall Monitoring: A Review**<br>
  *Natthapon Pannurat, Surapa Thiemjarus, and Ekawit Nantajeewarawat*
* [2] **Real-life/real-time elderly fall detection with a triaxial accelerometer**<br>
 *A. Sucerquia, J.D. López and J.F. Vargas-Bonilla*
* [3] **SisFall: A Fall and Movement Dataset**<br>
 *A. Sucerquia, J.D. López and J.F. Vargas-Bonilla*
* [4] **Fall-Detection Algorithm Using 3-Axis Acceleration: Combination with Simple Threshold and Hidden Markov Model**<br>
 *Dongha Lim, Chulho Park, Nam Ho Kim, Sang-Hoon Kim, and Yun Seop Yu*

![Image](https://i.imgur.com/yJieKKw.png)

-----------

# Dataset Information

## Activities of Daily Living (ADLs):


| Code | Activity | # Trials | Trial Length |
---|---|---|---
| D01  | Walking slowly | 1      | 100s     |
| D02  | Walking quickly | 1      | 100s     |
| D03  | Jogging slowly | 1      | 100s     |
| D04  | Jogging quickly | 1      | 100s     |
| D05  | Walking upstairs and downstairs slowly | 5      | 25s      |
| D06  | Walking upstairs and downstairs quickly | 5      | 25s      |
| D07  | Slowly sit in a half height chair, wait a moment, and up slowly | 5      | 12s      |
| D08  | Quickly sit in a half height chair, wait a moment, and up quickly | 5      | 12s|
| D09  | Slowly sit in a low height chair, wait a moment, and up slowly | 5      | 12s      |
| D10  | Quickly sit in a low height chair, wait a moment, and up quickly| 5      | 12s      |
| D11  | Sitting a moment, trying to get up, and collapse into a chair| 5      | 12s      |
| D12  | Sitting a moment, lying slowly, wait a moment, and sit again | 5      | 12s      |
| D13  | Sitting a moment, lying quickly, wait a moment, and sit again | 5      | 12s      |
| D14  | Being on oneís back change to lateral position, wait a moment, and change to oneís back  | 5      | 12s      |
| D15  | Standing, slowly bending at knees, and getting up | 5      | 12s      |
| D16  | Standing, slowly bending without bending knees, and getting up | 5      | 12s      |
| D17  | Standing, get into a car, remain seated and get out of the car | 5      | 25s      |
| D18  | Stumble while walking | 5      | 12s      |
| D19  | Gently jump without falling (trying to reach a high object)| 5      | 12s      |



-----------

## Falls:


| Code | Activity | # Trials | Trial Length |
---|---|--- |---
| F01  | Fall forward while walking caused by a slip| 5      | 15s      |
| F02  | Fall backward while walking caused by a slip| 5      | 15s      |
| F03  | Lateral fall while walking caused by a slip| 5      | 15s      |
| F04  | Fall forward while walking caused by a trip| 5      | 15s      |
| F05  | Fall forward while jogging caused by a trip| 5      | 15s      |
| F06  | Vertical fall while walking caused by fainting | 5      | 15s      |
| F07  | Fall while walking, with use of hands in a table to dampen fall, caused by fainting| 5      | 15s      |
| F08  | Fall forward when trying to get up| 5      | 15s      |
| F10  | Fall forward when trying to sit down | 5      | 15s      |
| F11  | Fall backward when trying to sit down | 5      | 15s      |
| F09  | Lateral fall when trying to get up | 5      | 15s      |
| F12  | Lateral fall when trying to sit down | 5      | 15s      |
| F13  | Fall forward while sitting, caused by fainting or falling asleep| 5      | 15s      |
| F14  | Fall backward while sitting, caused by fainting or falling asleep| 5      | 15s |
| F15  | Lateral fall while sitting, caused by fainting or falling asleep| 5      | 15s      |

-----------

## Subjects:


| Subject | Age | Height | Weight | Gender |      |     | Subject | Age | Height | Weight | Gender |
---------|-----|--------|--------|-------- |  --- |  ---|---------|-----|--------|--------|--------         
| SA01    | 26  | 165    | 53     | F      |      |     | SA13    | 22  | 157    | 55     | F      |
| SA02    | 23  | 176    | 58.5   | M      |      |     | SA14    | 27  | 160    | 46     | F      |
| SA03    | 19  | 156    | 48     | F      |      | | SA15    | 25  | 160    | 52     | F      |
| SA04    | 23  | 170    | 72     | M      |      | | SA16    | 20  | 169    | 61     | F      |
| SA05    | 22  | 172    | 69.5   | M      |      | | SA17    | 23  | 182    | 75     | M      |
| SA06    | 21  | 169    | 58     | M      |      | | SA18    | 23  | 181    | 73     | M      |
| SA07    | 21  | 156    | 63     | F      |      | | SA19    | 30  | 170    | 76     | M      |
| SA08    | 21  | 149    | 41.5   | F      |      | | SA20    | 30  | 150    | 42     | F      |
| SA09    | 24  | 165    | 64     | M      |      | | SA21    | 30  | 183    | 68     | M      |
| SA10    | 21  | 177    | 67     | M      |      | | SA22    | 19  | 158    | 50.5   | F      |
| SA11    | 19  | 170    | 80.5   | M      |      | | SA23    | 24  | 156    | 48     | F      |
| SA12    | 25  | 153    | 47     | F      |      |


-----------

## Correlations:

I have noticed that the average of the maximum obtained value of horizontal vector magnitude standard deviation per activity per subject is inversely proportional to the amplitude of the same measure in the first for falls. The correlation between the two aforementioned variables is -0.89. The first four ADLs are walking and jogging so I am trying to use them as a baseline measure for personal variance.

![Image](https://github.com/WJMatthew/SisFallAnalysis/blob/master/images/download%20(3).png)
<br>
![Image](https://github.com/WJMatthew/SisFallAnalysis/blob/master/images/download%20(2).png)
<br>

-----------

## Plotting method examples:
**plot_trials():** (Using set thresholds)
<br>
![Image](https://github.com/WJMatthew/SisFallAnalysis/blob/master/images/plot_trials_sample.png)
<br>
**plot_one_from_each():** <br>
![Image](https://github.com/WJMatthew/SisFallAnalysis/blob/master/images/plot_one_from_each_sample.png)
<br>
**plot_feats():** <br>
![Image](https://github.com/WJMatthew/SisFallAnalysis/blob/master/images/plot_feats_sample.png)
