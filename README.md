# IERP 2024 Baseline || <a href="http://www.iscslp2024.com/emotionChallenges">HOMEPAGE</a>

## Baseline model: Linear / GRU Attention Regressor

Participants predict the subjects' eight emotions(sadness, happiness, relaxation, surprise, fear, disgust, anger, neutral) and the intensity score of each emotion (range from 1 to 5 point for each emotion representing intensity, 1 indicating no such emotion, and 5 indicating the strongest emotion);

## Run: Sample 


```
python main.py --feature_set baichuan13B-base --fea_dim 5120 --epochs 200 --batch_size 128 --lr 0.0001 
```
We suggest that participants can first improve from the following perspectives
1. Use personality characteristics.
2. Feature fusion (early or late fusion).
3. When using visual features, it should be considered to integrate the two feature files of "watch" and "description".

If you have any questions, you can contact us through the official email：IEPR2024@iscslp2024.com
