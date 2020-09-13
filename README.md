# NBA_Machine_Learning
A machine learning model to predict NBA 2k data
# Summary

**Goal**: 
To implement web scraping to extract data and then evaluate both supervised (```Linear Regression```, ```Logistic Regression```) and unsupervised machine learning classifiers (```KMeans Clustering```) in their success in predicting NBA 2K video game scores the deaths of Game of Thrones characters using data from the book series.

You can find a Medium post I wrote summarizing the web scraping portion of this project at this link: https://towardsdatascience.com/web-scraping-nba-2k-data-d7fdd4c8898c. 

## Dataset

I web scraped data from basketball-reference.com (www.basketball-reference.com/leagues/NBA_2014_totals.html) and hoopshype.com (https://hoopshype.com/2015/11/05/these-are-the-ratings-of-all-players-in-nba-2k16/) to extract real life NBA and NBA 2k video game data between 2014 and 2016.

### Preprocessing

![Web Scraping](https://miro.medium.com/max/1400/1*mldbTJKcExM6zH7B9BSohQ.png)

After scraping the NBA 2k ratings to a JSON format, I joined the data with the metrics of the real life NBA players. By right joining to avoid missing values, I only kept players that were included in the 2K game.

## Models

### Regressions

Both linear and logistic regression models gave similar root mean square errors of *7.57* and *7.76* respectively. The linear regression model gave a *model score of 0.90* while logistic regression gave a much lower model score. This indicates a linear problem set with high correlation between metrics and 2K rating.

### Clustering

A KMeans Clustering model was used to cluster positions and explore if players truly made plays according to their position or if being a generalist was common. The optimal number of clusters was determined to be either 4 or 5, which is similar to the 5 core positions the NBA employs. 

The results of the clustering analysis revealed significant intermixing between clusters when it cames to steals and defensive maneuvers, but less mixing for 2 and 3 pointers made. This might indicate less rigid defensive responsibilities for players.


