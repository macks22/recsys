# Load jester dataset ratings
ratings = scan('data/ratings.txt',sep='\n')

png(filename="data/figs/ratings-density-plot")
plot(density(ratings),main="Jester Rating Distribution Density")
dev.off()

png(filename="data/figs/ratings-histogram")
hist(ratings,main="Jester Rating Distribution Histogram")
dev.off()

descdist(ratings)

# Estimate parameters for beta distribution to fit to data
estBetaParams <- function(mu, var) {
  alpha <- ((1 - mu) / var - 1 / mu) * mu ^ 2
  beta <- alpha * (1 / mu - 1)
  return(params = list(alpha = alpha, beta = beta))
}

# find params and get quantiles for Q-Q plot
scaledRatings = (ratings - min(ratings)) / diff(range(ratings))
params = estBetaParams(scaledRatings)
n = length(ratings)
probs = (1:n)/(n+1)
betaQuants = qbeta(probs, shape1=params$alpha, shape2=params$beta)

# Finally plot the theoretical vs. empirical on q-q plot for comparison
plot(sort(betaQuants), sort(scaledRatings),
     xlab="Theoretical Quantiles for Beta Dist.",
     ylab="Sample Quantiles: Jester Joke Ratings",
     main="Beta Q-Q Plot of Jester Joke Ratings")
abline(0,1)
