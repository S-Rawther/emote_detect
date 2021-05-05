# Emotion Detection

A Basic Human Emotion Detecting Algorithm that can classify a face into any of seven expression. The Data is trained using **FER-2013** Dataset. The Dataset consists of Seven Different Emotions namely - 
**Angry, Disgust, Fear, Happy, Sad, Surprise & Neutral**

## Requirements ##
* Python 3
* OpenCV
* NumPy
* Tensorflow
* Keras
* Adam
* Pandas

You can install the required dependencies by running:
```
pip install -r packages.txt
```

## Installation ##

If you're using pycharm or any similar IDE, then you can just download and extract the zip file straight away. If not, then clone the repo using:
```bash
git clone https://github.com/Shahin-Nishad/emote_detect.git
cd emote_detect
python train_AI.py
```

**You can Download FER2013 Dataset CSV file from:**
[FER2013] (https://www.kaggle.com/deadskull7/fer2013)

Once the training is done, you can run the program by:

```bash
cd emote_detect
python emote_detect.py
```

## Algorithm ##

The AI works by using Haar Cascade to identify face from each frame of the capture device, which is then cropped to **48 pixel square** for the Neural Network. The Output is decided using the most probable emotion from the list of Softmax Score
