setwd("~/PycharmProjects/CryptoTrading")
library(ggplot2)

# Functions
{
read = function(coin_type, nulls, minLO) {
  d  = read.csv(paste('Training_Data/', coin_type, '_', nulls, '.csv', sep=''))
  d = subset(d, d$minutes > minLO)
  d$weekday = factor(d$weekday)
  d$hour = factor(d$hour)
  d$X = as.POSIXct(d$X, format='%Y-%m-%d %H:%M:%S')
  rownames(d) = d$X
  print(paste(nrow(d), 'observations meeting minimum tweet limit'))
  return(d)
}
corr = function(data=d, coin_type) {
  library('midasr')
  data$prev_return = mls(data$return, 1, 1)
  lim = 100 * max(abs(data$return))
  corr_data = data[complete.cases(data), ]
  corr_val = cor(corr_data$return, corr_data$prev_return)
  
  ggplot(data=data, aes(x=prev_return * 100, y=return * 100)) + 
    geom_point(color='skyblue') + geom_hline(yintercept=0, linetype='dashed', alpha=0.5) +
    geom_vline(xintercept=0, linetype='dashed', alpha=0.5) +
    labs(x = 'Return on hour t (%)', y='Return on hour t+1 (%)',
         title=paste('Correlation =', round(corr_val, 4)))
}
count_mag = function(data=d, normed) {
  y = 100 * abs(data$return)
  if (normed == T) {
    x = data$normed.count
  } else {
    x = data$count
  }

  plot(x, y, type='p', col='blue', pch=16, xlab='# of Tweets in previous hour', ylab='Magnitude of prie change (%)')
  abline(lm(y~x), lty=3, lwd=2, col='red')
}
sent_dir = function(data=d, clean=F, normed=T) {
  library(pals)
  if (clean == T) {
    x = data$clean.neg
    y = data$clean.pos
  } else {
    x = data$raw.neg
    y = data$raw.pos
  }

  ggplot(data=data, aes(x=x, y=y)) + 
    geom_point(aes(color=100*data$return), size=2) +
    scale_color_gradientn('Hourly Return\nat hour t+1 (%)', colors=coolwarm(400), breaks=c(-5.0, -2.5, 0.0, 2.5, 5.0)) + 
    labs(x='Tweet Negativity at hour t', y='Tweet Positivity at hour t')
}
sent_mag = function(data=d, clean=F) {
  library(viridis)
  if (clean == T) {
    x = data$clean.sub
    y = data$clean.pol
  } else {
    x = data$raw.sub
    y = data$raw.pol
  }
  
  ggplot(data=data, aes(x=x, y=y)) + 
    geom_point(aes(color=100*abs(data$return)), size=2) + 
    scale_color_viridis(option='plasma') +
    labs(x = 'Tweet Subjectivity at hour t', y='Tweet Polarity at hour t', color='Magnitude of\nHourly Return\nat hour t+1(%)')
}
linreg = function(data=d, type='all', clean=F, count=T) {
  
  # Select feature vectors
  if (type == 'blob') {
    if (clean == T) {
      df = subset(data, select=c(blob_clean_pol, blob_clean_sub))
    } else {
      X = df = subset(data, select=c(blob_raw_pol, blob_raw_sub))
    }
  } else if (type == 'vader') {
    if (clean == T) {
      df = subset(data, select=c(vader_clean_pos, vader_clean_neu, vader_clean_neg))
    } else {
      df = subset(data, select=c(vader_raw_pos, vader_raw_neu, vader_raw_neg))
    }
  } else {
    if (clean == T) {
      df = subset(data, select=c(vader_clean_pos, vader_clean_neu, vader_clean_neg, blob_clean_pol, blob_clean_sub))
    } else {
      df = subset(data, select=c(vader_raw_pos, vader_raw_neu, vader_raw_neg, blob_raw_pol, blob_raw_sub))
    }
  }
  
  if (count == T) {
    df = merge(df, subset(data, select=c(count)), by=0)
    rownames(df) = df$Row.names
    df = subset(df, select=-c(Row.names))
  }
  
  # add y vector
  df = merge(df, subset(data, select=c(return)), by=0)
  rownames(df) = df$Row.names
  df = subset(df, select=-c(Row.names))
  
  # Perform fit
  fit = lm(abs(return) ~., data=df)
  summary(fit)
}
bisect = function(f, a, b, num=10, eps=1e-5) {
  c = (a + b) / 2
  while(f(c) !=0 && b-a > 0.00002) {
    if(f(c) == 0) {
      c
    }
    if(f(c) <0 ) {
      a=c
    }
    else {
      b=c
    }
    
  c = (a + b) / 2
  return(c)
  }
}
geom_mean=function(data=dret){
  library(psych)
  data=dret
  pos = subset(data$dr/100, data$move == 1)
  neg = subset(-data$dr/100, data$move == -1)
  all = rbind(pos, neg)
  
  p_avg = 100 * (geometric.mean(1 + pos) - 1)
  n_avg = 100 * (geometric.mean(1 + neg) - 1)
  all_avg = 100 * (geometric.mean(1 + all) - 1)
  
  print(paste('Average Positive Return: ', round(p_avg, 3), '% for ', length(pos), ' hours.'))
  print(paste('Average Negative Return: ', round(n_avg, 3), '% for ', length(neg), ' hours.'))
  print(paste('Average Total Return: ', round(all_avg, 3), '% for ', length(all), ' hours.'))
  
}

min_acc = function(days, c0=1000, data=d) {
  library(psych)
  pos = subset(data, data$move == 1)
  neg = subset(data, data$move < 1)
  trade_fee = c(0, 0.0026, 0.0024, 0.0022, 0.0020, 0.0018, 0.0016, 0.0014, 0.0012, 0.0010)
  trade_vol0 = c(0, 50, 100, 250, 500, 1000, 2500, 5000, 10000, 20000)*1000/c0
  
  trades = days*12
  trade_vol = subset(trade_vol0, trade_vol0 < trades)
  trade_vol = append(trade_vol, trades-tail(trade_vol, 1))
  
  p_avg = 1 + geometric.mean(pos$return)
  n_avg = 1 + geometric.mean(-neg$return)
  
  f = function(x, p=p_avg, n=n_avg, fee=trade_fee, vol=trade_vol) {
    fun = 1
    for (i in 2:length(trade_vol)) {
      fun = fun * (((p-fee[i])*(n-fee[i])*p*n)^(x/2) * ((2-n-fee[i])*(2-p-fee[i])*(2-n)*(2-p))^((1-x)/2)) ^ (vol[i] - vol[i-1])
    }
    
    return(fun - 1)  
  }
  
  
  min_acc = bisect(f, 0, 1)
  sprintf('Minimum accuracy required for profitability is : %.3f', min_acc)
}
ret_spread = function(filename='bitcoin price history study.xlsx') {
  library(midasr)
  library(xlsx)
  dret = read.xlsx(paste('Historical_Data/', filename, sep=''), sheetIndex=1)
  r = dret$close
  r = r/r[1]
  ret = 100 * (r - mls(r, 1, 1))
  dret$dr = ret
  dret$move = sign(dret$dr)
  return(dret)
}
res_acc = function(filename){
  results = read.csv(paste('Saved_Models/', filename, '.csv', sep=''))
  results$correct = ifelse(results$actual == results$preds, 'Correct', 'Incorrect')
  results$correct = factor(results$correct)
  results$bool = ifelse(results$actual == results$preds, F, T)
  boolColors <- as.character(c("TRUE"="darkgreen", "FALSE"="firebrick"))
  boolScale <- scale_colour_manual(name="bool", values=boolColors)
  ggplot(data=results, aes(x=correct, y=100*abs(ret), color=bool)) + geom_jitter(size=0.5) + geom_boxplot(alpha=0.5, outlier.color=NA) + 
    labs(x='Prediction Result', y='Magnitude of Hourly Return (%)') + boolScale + theme(legend.position="none")
}
}



# Settings
coin = 'bitcoin'
null_vals = 'all'
min_bound = 50

# Prepare Data
d = read(coin, null_vals, min_bound)

# ----------------------------------------------------------------------------------------------#

# Tweets
ggplot(data=d, aes(x=X)) + geom_col(aes(y=count), fill='lightskyblue', alpha=0.9) + labs(x = 'Hours of Collection') + labs(y='Number of Tweets') + 
  ggtitle(paste(round(sum(d$count)*1e-6, 2), 'million Tweets Total')) + scale_x_datetime(date_breaks='4 day', date_labels='%b %d')
ggsave("Figures/Collection_Period.png", units="in", width=12, height=5, dpi=300)

# Tweets by time
ggplot(d, aes(x=hour, y=count)) + stat_summary(fun.y="mean", geom="bar", fill='lightskyblue', color='white') + labs(x='Hour of the Day', y='Averae Number of Tweets')

ggplot(d, aes(x=weekday, y=count)) + stat_summary(fun.y="mean", geom="bar", fill='lightskyblue', color='white') + labs(x='Day of the Week', y='Averae Number of Tweets')

d_counts = aggregate(count ~ hour + weekday, data=d, FUN='mean')
ggplot(data = d_counts, aes(x=hour, y=count, fill=weekday)) + geom_bar(stat='identity') + labs(x='Hour of the Day (UTC)', y='Average Number of Tweets', fill='Day of the Week' ) + scale_fill_discrete(labels=c('Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'))
ggsave("Figures/Tweet_Count.png", units="in", width=8, height=4, dpi=300)


# Time correlation
corr(d, coin)
ggsave("Figures/Correlation.png", units="in", width=6, height=5, dpi=300)

# Sentiment
d$raw.sent = d$raw.pos - d$raw.neg
d$clean.sent = d$clean.pos - d$clean.neg


ggplot(data=d, aes(d$raw.sent)) + geom_histogram(bins=100, col="darkgrey", fill="skyblue", alpha=0.6) + geom_density(col='darkgrey') + 
  labs(title="Histogram of Sentinement Analysis Results", x="Average Tweet Sentiment", y="Count")


# Return Magnitude
count_mag(d, normed=T)
sent_mag(d, clean=F)
ggsave("Figures/Sentiment_Magnitude.png", units="in", width=6, height=5, dpi=300)
# Return Amount
sent_dir(d, clean=F)
ggsave("Figures/Sentiment_Return.png", units="in", width=6, height=5, dpi=300)

# Linear Regression
linreg(type='blob', clean=F, count=T)

# Minimum Accuracy required for Positive Returns
min_acc(data=d, days=100, c0=1000)

# Bitcoin Pricing History
dret = ret_spread()
dret = dret[!is.na(dret$dr), ]
ret = dret$dr

# Hourly Return Histogram
ggplot(dret, aes(dr)) + geom_histogram(aes(ret), bins=150, fill="skyblue", alpha=0.6) + 
  labs(x="Hourly Return (%)", y="Count")
ggsave("Figures/Hourly_Return_Hist.png", units="in", width=12, height=5, dpi=300)

geom_mean()

# Hourly Return ECDF
ggplot(dret, aes(dr)) + stat_ecdf(geom='point') + geom_hline(yintercept=0, color='darkgrey', size=0.5, linetype=2) +
  geom_hline(yintercept=1, color='darkgrey', size=0.5, linetype=2) + labs(x='Hourly Returns (%)', y='Fn(Daily Retrns)')
ggsave("Figures/Hourly_Return_ECDF.png", units="in", width=6, height=6, dpi=300)

# Hourly Return qq-Plot
ggplot(dret, aes(sample=dr)) + geom_point(stat='qq',  show.legend = F) + labs(x='Theoretical Quantiles', y='Sample Quantiles') + geom_abline(aes(intercept = 0, slope = 1, color='Normal Distribution'), linetype=2, size=0.7) + guides(name='', color=guide_legend(override.aes=list(linetype=2))) + scale_color_manual(name="", values=c("firebrick"))+ theme(legend.position = c(0.2, 0.9), legend.background = element_rect(fill=F), legend.text=element_text(size=13))
ggsave("Figures/Hourly_Return_QQ.png", units="in", width=6, height=6, dpi=300)

(max(ret) - mean(ret)) / sd(ret)

# Accuracy of Results
res_acc('Model Results 57_3')
ggsave("Figures/result_acc.png", units="in", width=6, height=6, dpi=300)
