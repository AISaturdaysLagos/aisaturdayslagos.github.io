# Recommender

Recommender systems are ubiquitous in many online sites.  If you've ever been recommended an item to buy, a movie to watch, or a person to follow on social media, you have seen the results of a recommender system.  The goal of recommender systems, broadly speaking, is to suggest items to a user that the system believes this user will like.  This has obvious applications in many online stores, where both the company and the users will likely be happier if the system is able to provide high-quality recommendations.

Broadly speaking, there are a number of sources of information we can use to make predictions about whether a user will like an item.  There is "pure user" information, such as the age, location, profession, etc, of the user; these don't involve any of the items themselves, but just contain information about the user.  There is also "pure item" information (not related to any user, but just to the item); in the case of movies, for instance, this could include things like the movie budget, the genre, the actors, etc.  Finally, there is "user-item" information, which consists of ratings or order history between users and items.  Ideally, we would want to use all of this information to provide as high-quality recommendations as possible to the user.

Although there are techniques that combine all these types of data, in this set of notes we are largely going to focus just on the third type, building recommendation systems that rely only on past user-item information.  This is a setting known as _collaborative filtering_, and these algorithms will form the basis of the presentation here.

Before we begin our discussion, however, we should emphasize that there are a number of issues that we won't touch on in these notes.  Here we will consider a standard setting where each user "rates" some (usually quite small) some subset of the items.  This is the classical recommender system setting, and forms the intuitive basis for much of the algorithmic work in this area.  However, in reality there are many challenges that arise in this setting that need to be dealt with in actual real-world recommender systems.  For example, users may rarely actually rate items, but instead the mere presence of a user-item order will be the only information we have (so we have no formal indication of the user rating, but we can typically assume that the user orders items they tend to like).  Additionally, when a new user signs up, or a new item is added, there is initially user-item history, so we likely need to integrate some of the user or item-level features in order to form "baseline" predictions in these cases; and in general, there is a question of how much "personalization" we want to offer versus how much we should recommend "generic" good items.



## Collaborative filtering

_Collaborative filtering_ is one way to build recommendation systems, that (at least by the definitions we will use here), only users user-item information in order to make recommendations.  The setting you should have in mind here is that of a matrix, where the rows of the matrix correspond to users, and the columns of the matrix correspond to items.  The entires of the matrix should be though of as ratings for each item, i.e., how much user $i$ with like item $j$.  For example, consider the following matrix

$$
X \in \mathbb{R}^{m \times n} = \left [ \begin{array}{cccc}
1 & 1 & 2 & 3 \\
2 & 2 & 5 & 4 \\
2 & 3 & 3 & 5 \\
4 & 2 & 4 & 5 \\
2 & 2 & 4 & 4
\end{array} \right ]
$$


For example, in this setting user 1 would assign a ratings of 1, 1, 2, 3 to items 1-4 respectively (i.e., the first row of the matrix).  Similar, item 2 has ratings of 1,2,3,2,2 from users 1-5.  The ordering of this matrix is important (rows correspond to users and items correspond to columns), so it's important to be familiar with this notation; note that in this convention there are $m$ users and $n$ items.

Of course, in general, we _won't_ have ratings for every user for every item (this is precisely the problem we're trying to solve).  Rather, we only observe some sparse subset of the entries of $X$, whereas the rest are unknown (denoted by $?$ in the matrix below).

$$
X = \left [ \begin{array}{cccc}
1 & ? & ? & 3 \\
? & 2 & 5 & ? \\
? & 3 & ? & 5 \\
4 & ? & 4 & ? \\
2 & 2 & ? & ?
\end{array} \right ]
$$

The task of collaborative filtering, then, is to "fill in" the remaining entries of this matrix given the observed matrix.  This $X$ matrix that we observed is _sparse_ but the unknown entries do not correspond to actual zeros in the matrix, but are rather just truly unknown.

In the collaborative filtering setting, we want to fill in the remaining entries from the matrix based _only_ upon the observed entries (that is, we don't have additional context such as features for the user or items, but as mentioned above, in typical recommender systems, you _do_ need to consider these other options.

### Example: MovieLens data set

The example we'll use in these notes (and which you'll use in the homework), is the MovieLens data set, a common publicly available data set for collaborative filtering.  The full data set and information is available here: https://grouplens.org/datasets/movielens/.  Although the full MovieLens data set is fairly large, there is a smaller subset available here: http://files.grouplens.org/datasets/movielens/ml-latest-small.zip which we'll use for this example.  We can load the data using the following calls (importantly, this is slightly different than you'll need to implement for the homework, but it's useful for our discussion here).

```python
# load MovieLens ratings, keeping only movies with >=5 ratings
df = pd.read_csv("ratings.csv")
df_mov = pd.read_csv("movies.csv", index_col="movieId")
X = np.asarray(sp.coo_matrix((df["rating"], (df["userId"]-1, df["movieId"]-1))).todense())
```

Note that we're generating a dense matrix of user IDs by movie IDs, and for larger systems you would definitely need to keep the matrix in sparse form.  While this data set was already filtered to only include users who ranked a reasonable number of movies, we're also going to filter it to only include movies with five or more ratings.

```python
valid_movies = (X!=0).sum(axis=0) >= 5
movie_to_title = dict(zip(range(len(valid_movies)), df_mov.loc[np.where(valid_movies)[0]+1]["title"]))
X = X[:,valid_movies]
print(X.shape)
```

<pre>
(671, 3496)

</pre>


Note that in this format, the zero entries correspond to unknown elements, not to actual zero ratings (the ratings only go as low as 0.5).  For instance, we can compute per-user and per-item mean ratings via the following.

```python
# form average rating for each user
user_means = np.array([X[i,X[i,:]!=0].mean() for i in range(X.shape[0])])
movie_means = np.array([X[X[:,i]!=0,i].mean() for i in range(X.shape[1])])
```

Finally, we can get some basic sense of this dataset by looking at average ratings and average numbers of ratings.

```python
plt.hist(user_means, bins=20)
plt.xlabel("Mean User Rating")
plt.ylabel("Count")
```

{% include image.html img="output_0.svg" %}

```python
plt.hist(movie_means, bins=20)
plt.xlabel("Mean Movie Rating")
plt.ylabel("Count")
```

{% include image.html img="output_1.svg" %}

```python
num_reviews = (X!=0).astype(float).sum(axis=0)
plt.hist(num_reviews,bins=np.logspace(np.log10(min(num_reviews)),np.log10(max(num_reviews)),10))
plt.gca().set_xscale("log")
plt.xlabel("Number of Reviews")
plt.ylabel("Number of Movies")
```

{% include image.html img="output_2.svg" %}

```python
num_reviews = (X!=0).astype(float).sum(axis=1)
plt.hist(num_reviews,bins=np.logspace(np.log10(min(num_reviews)),np.log10(max(num_reviews)),10))
plt.gca().set_xscale("log")
plt.xlabel("Number of Reviews")
plt.ylabel("Number of Users")
```

{% include image.html img="output_3.svg" %}

Finally, since I know you're likely curious, here are the best and worst reviewed movies.

```python
# best movies
print("\n".join([movie_to_title[a] for a in np.argsort(movie_means)[-5:]]))
```

<pre>
Love &amp; Human Remains (1993)
Diabolique (Les diaboliques) (1955)
Paperman (2012)
Ikiru (1952)
Anne Frank Remembered (1995)

</pre>


```python
# worst movies
print("\n".join([movie_to_title[a] for a in np.argsort(movie_means)[:5]]))
```

<pre>
Joe&#x27;s Apartment (1996)
After Earth (2013)
Battlefield Earth (2000)
Max Payne (2008)
Cats &amp; Dogs (2001)

</pre>


### Historical note: the Netflix Prize

The field of collaborative filtering received a substantial boost in popularity during the late 2000s after the annoucement of the Netflix Prize.  This was a public competition where Netflix released a large dataset of about 100 million ratings, and challenged the public to come up with a method that improved, by 10% according to RMSE, upon their existing solution (which relied upon item-item Pearson correlation, plus some linear regression methods).

Though there were some notable issues with the Netflix Prize, the competition had a huge impact on the field of data science.  Although there were many prior data competitions of a similar form (the KDD conference had been running the KDDCup for many years), this was the competition that really put "machine learning competitions" into the public eye.  Companies like Kaggle were started largely due to the success of this competition, and many later contests like the ImageNet challenge were likely directly or indirectly influenced by it.  Furthermore, the ubiquity of ensemble methods (not as in algorithm like boosting or bagging, but as in combining many disparate models to form overall better predictions) was heavily influenced by the competition, where all the top teams used ensemble methods.  Finally, it was an amazing stroke of luck that the target of 10% improvement seemed to be almost _exactly_ what was possible combining a huge variety of methods; the competition could have easily taken a few months or still be running today if the data had offered only a slightly lower or slightly higher possible level of performance.

### Types of collaborative filtering algorithms

While there are several possible algorithms for collaborative filtering, broadly speaking most of them fit into one of three different categories:

- **User-user approaches:**  In this approach we estimate a user's rating of an item by finding "similar" users and then looking at their predictions for this item.
- **Item-item approaches:** These methods take the converse approach, and estimate a user's rating of a item by finding similar items and then looking at the user's rating of these similar items.
- **Matrix factorization**: Finally, the last class of approaches works a little bit differently, by aiming to construct a _low-rank_ matrix that approximates the observed entries of the rating matrix.

We'll discuss each of these in turn, and look at how they can be implemented on the MovieLens data set (except for matrix factorization, which is one of the homework assignments).

## User-user methods

As mentioned above, the key idea of the user-user approach is that in order to predict an unknown user-item score, we're going to find "similar" users, and then look at how they scored that item.  To compute similar users, we can use common techniques like correlation coefficient, cosine_similarity, etc, with the caveat that these metrics typically assume fixed-sized vectors with no unknown elements, so some changes are typically needed to apply to the collaborative filtering setting.

To start, let's introduce a slightly more formal bit of notation to define our problem.  Let $\hat{X}\_{ij}$ denote our prediction for the $i$th user and $j$th item (i.e., this will be one of the elements that is missing from the matrix $X$, which we want to predict).  A common form for the prediction make by the user-user approach would be

$$
\hat{X}_{ij} = \bar{x}_i + \frac{\sum_{k:X_{kj} \neq 0} w_{ik} (X_{kj} - \bar{x}_k)}{\sum_{k:X_{kj} \neq 0} \lvert w_{ik} \rvert}
$$

where $\bar{x}\_i$ denotes the average of the observed ratings for user $i$, and $w\_{ik}$ denotes a _similarity weight_ between user $i$ and user $k$ (which we will define shortly).  The intuition behind this approach is the following: if we want to predict user $i$'s rating for item $j$, we look across all users that _do_ have ratings for item $j$, and we average these together, weighted by a similarity function between the two users (we divide by $\sum\_{k:X\_{kj} \neq 0} \lvert w\_{ik} \rvert$ so that we are taking a weighted average, noting that we take the absolute value because similarity weights can sometimes be positive or negative depending how we define then).  Because user's also frequently have their own "baseline" rating (i.e., some users naturally assign lower ratings than others), it's slightly better to do this modeling in the "difference space", the difference between a user's rating and their mean rating, and then add re-scale by adding a user's mean score.

Let's see how this works in code.  For now, let's suppose we have all the user/user weights in the $W \in \mathbb{R}^{m \times m}$ (remember, by our convention there are $m$ users), which we'll discuss more shortly.  But assuming we have these weights, we can make a prediction by the following code.

```python
def predict_user_user(X, W, user_means, i, j):
    """ Return prediction of X_(ij). """
    return user_means[i] + (np.sum((X[:,j] - user_means) * (X[:,j] != 0) * W[i,:]) / 
                            np.sum((X[:,j] != 0) * np.abs(W[i,:])))
```

Assuming a similarity matrix of all ones, let's see how the system predict that user 0 and user 10 would score two different movies.

```python
W = np.ones((X.shape[0], X.shape[0]))
print("User: 0, ", movie_to_title[0], predict_user_user(X, W, user_means, 0, 0))
print("User: 10, ", movie_to_title[0], predict_user_user(X, W, user_means, 10, 0))
print("User: 0, ", movie_to_title[1], predict_user_user(X, W, user_means, 0, 1))
print("User: 10, ", movie_to_title[1], predict_user_user(X, W, user_means, 10, 1))
```

<pre>
User: 0,  Toy Story (1995) 2.76182595231
User: 10,  Toy Story (1995) 4.27064948172
User: 0,  Jumanji (1995) 2.37956820083
User: 10,  Jumanji (1995) 3.88839173024

</pre>


Since computing predictions one at a time is fairly slow, let's see how we can form all the predictions for a user simultaneously (note that we add a small constant to the denominator to cover the case where there is no user (with non-zero weight), who has previously rated the item.

```python
def predict_user_user(X, W, user_means, i):
    """ Return prediction of X_(ij). """
    return user_means[i] + (np.sum((X - user_means[:,None]) * (X != 0) * W[i,:,None], axis=0) / 
                            (np.sum((X != 0) * np.abs(W[i,:,None]), axis=0) + 1e-12))
```

```python
plt.hist(predict_user_user(X, W, user_means, 0), bins=20)
```

{% include image.html img="output_4.svg" %}

### Computing weights

Of course (since the weights are all equal) we aren't doing too much here, mainly just computing average scores based upon a user's mean and the average rating that other reviewers have assigned to a movie.  Instead, for this method to be effective, we want to compute a weight matrix that somehow captures the similarity between two users.  Because the only information we have in collaborative filtering is based upon the user-item ratings (i.e., we don't have user-centric features that could be used to compute a similarity between users), the main way that we can build a similiarty score is by looking at some kind of correlation between the two sets of ratings; but since two users haven't necessarily rated the same movies, it is common to compute these correlations _only for the set of movies that both have reviewed_.  We'll consider a few examples of such weighting schemes.

**Pearson correlation** Let's take the standard example of Pearson correlation (one of the most common methods for defining these weights).  First, let $\mathcal{I}\_{ik}$ be the set of movies that user $i$ and user $k$ both rated

$$
\mathcal{I}_{ik} = \{ j : X_{ij} \neq 0, X_{kj} \neq 0 \}
$$

The Pearson correlation between two users gives the weight

$$
W_{ik} = \frac{\sum_{j \in \mathcal{I}_{ij}} (X_{ij} - \bar{x}_i)(X_{kj} - \bar{x}_k)}
{\sqrt{\sum_{j \in \mathcal{I}_{ij}}(X_{ij} - \bar{x}_i)^2} \sqrt{\sum_{j \in \mathcal{I}_{ij}}(X_{kj} - \bar{x}_k)^2}}
$$

which is just the normal Pearson correlation coefficient but only computed over the items where both users rated the item.

The Pearson correlation can be computed via the following function (we add $10^{-12}$ to the denominator to not cause the result to be undefined when the users have no overlap in items).

```python
def pearson(X,user_means, i,j):
    I = (X[i,:]!=0) * (X[j,:]!=0)
    xi = X[i,I] - user_means[i]
    xj = X[j,I] - user_means[j]
    return (xi @ xj)/(np.sqrt((xi @ xi)*(xj @ xj))+1e-12)
pearson(X,user_means, 1, 2)
```

However, as before, it's inefficient to compute these individually for each pair of users.  The following function computes the same quantity for all users via the standard matrix tricks.

```python
def all_pearson(X,user_means):
    X_norm = (X - user_means[:,None])*(X != 0)
    X_col_norm = (X_norm**2) @ (X_norm != 0).T
    return (X_norm @ X_norm.T)/(np.sqrt(X_col_norm*X_col_norm.T)+1e-12)
W_pearson = all_pearson(X, user_means)
W[1,2]
```

One downside to the standard pearson correlation as defined above is that if users have a very small number of items in common (often, just 1), then their Pearson correlation is either 1 or -1.  This is almost certainly not what is wanted, so it's common to require some minimum number of items in common, and set the weight to zero otherwise.

```python
def all_pearson(X, user_means, min_common_items=5):
    X_norm = (X - user_means[:,None])*(X != 0)
    X_col_norm = (X_norm**2) @ (X_norm != 0).T
    common_items = (X!=0).astype(float) @ (X!=0).T
    return (X_norm @ X_norm.T)/(np.sqrt(X_col_norm*X_col_norm.T)+1e-12) * (common_items >= min_common_items)
W_pearson = all_pearson(X, user_means)
W[1,2]
```

**Cosine similarity** The standard cosine similarity is also fairly common to use as a weight, and differs in that (under the usual definition), we _don't_ subtract off the mean before computing the similarity.  It is defined as follows

$$
W_{ik} = \frac{\sum_{j=1}^n X_{ij}X_{kj}}{\sqrt{\sum_{j=1}^n X_{ij}^2} \sqrt{\sum_{j=1}^n X_{kj}^2}}
$$

Notice that (again, in the common definition of cosine similarity for collaborative filtering) we also don't explicitly sum over only the elements where the the user both rated an item.  Because we treat missing entries in the matrix as zeros here, these terms won't factor in to the numerator (any items where both users did not rate an item will be zero), but the denominator _is_ affected by the total number of items each user rated.  Thus, if two users both rated a large number of items but their overlap is small, the cosine similarity would be smaller than in the case where they both rated a small number of items that have the number number of overlaping items.  Thus, we usually don't need to explicitly filter for users that have some minimum number of items in common, as we did with Pearson correlation.

The following code will compute the cosine similarity (this time just directly adopting the setting of computing all similarities simultaneously).

```python
def all_cosine(X):
    x_norm = np.sqrt((X**2).sum(axis=1))
    return (X @ X.T) / np.outer(x_norm, x_norm)
W_cosine = all_cosine(X)
```

**Nearest neighbors** After computing one of the above similarity measures (or other similar measures), it's fairly common to look _only_ at some $K$ users that are most similar to a given user, rather than a sum over all users.  This is equivalent to setting the lower entries of $W$ to all be zero, i.e.,

$$
\tilde{W}_{ik} = \left \{ \begin{array}{ll}W_{ik} & \mbox{ if $\lvert W\_{ik} \rvert$ is amongst top $K$ values for user $i$} \\
0 & \mbox{ otherwise} \end{array} \right .
$$

Doing so effectively looks just at the most similar users (i.e., the "nearest neighbors") of user $i$, instead of all other users.  The downside of this approach is that if our data matrix is very sparse, we may not find _any_ other users that have ratings for some new item $i$, based upon only the neareast neighbors.  This could be fixed by considering only those nearest neighbors that _do_ have a given item, but this is computationally more expensive, since we need to recompute neighbors for each item, instead of once per user.

For any existing weight matrix we can limit it to the nearest neightbors with the following function.

```python
def nearest_neighbors(W, K):
    W_neighbors = W.copy()
    W_neighbors[W < np.percentile(np.abs(W), 100-100*K/X.shape[0], axis=0)] = 0
    return W_neighbors
W_pearson_100 = nearest_neighbors(W_pearson, 100)
```

Because it's part of the homework (at least for the case of matrix factorization), we're not going to rigorously evaluate the performance of the approach, but we can apply the approaches by just plugging in these weights into the functions we considered above.

```python
plt.hist(predict_user_user(X, W_pearson, user_means, 0), bins=20)
```

{% include image.html img="output_5.svg" %}

Using the Pearson correlation produces a great deal more spread in the prediction than in our previous case where we had unit weights.  Note that beacuse the weights can be negative, and we're modeling a difference, it's completely possible to get ratings outside the allowable range (in practice, we'd simply clip them in this case).  Let's look at the case where we use just the 100 nearest neighbors to make the prediction.

```python
plt.hist(predict_user_user(X, W_pearson_100, user_means, 0), bins=20)
```

{% include image.html img="output_6.svg" %}

What's happening here is that if we restrict ourselves to only the 100 nearest neighbors of a given user $i$, then there are many cases were we don't have any overlap with user who predict the for some item, so we make the mean prediction much more (this is intentially a user for this this is a large difference, other users have less difference, if they happen to have neighbors that rate a more diverse set of items).  Finally, let's look at the cosine similarity-based predictions.

```python
plt.hist(predict_user_user(X, W_cosine, user_means, 0), bins=20)
```

{% include image.html img="output_7.svg" %}

## Item-item approaches

We'll mention item-item approaches only very briefly here, because the math is essentially identical to the user-user case, just with the matrices transposed.  Whereas the user-user approach was based upon which _users_ were similar to a given user, the item-item approach is based upon what _items_ are similar to a given item.  More formally, to make a prediction about the likely rating for user $i$ and item $j$, we form a weighted combination of the other _item ratings_ that user $i$ has specific, weighted by the similarity between the two items.  Formally, this is described by the equation


$$
\hat{X}_{ij} = \bar{x}_j + \frac{\sum_{k:X_{ik} \neq 0} w_{jk} (X_{ik} - \bar{x}_k)}{\sum_{k:X_{ik} \neq 0} \lvert w_{jk} \rvert}
$$


where here $\bar{x}\_j$ denotes the average rating for _item j_ (not per-user, as it was before) and $w\_{jk}$ denotes the similarity beween item $j$ and item $k$.  Notice that here $k$ sums over other items, not over other users.  The weights can be similarly defined by Pearson correlations or cosine similarity, but again here considering the similarity between items instead of users.

One nice this above the approach is that we can reuse all the exact same functions as before, just transposing the relevant matrices.

```python
def predict_item_item(X, W, item_means, i):
    return predict_user_user(X.T, W, item_means, i)

W_pearson = all_pearson(X.T, movie_means)
```

Note that to get a list of all user predictions, we'd need how to iterate over all items (i.e., form the entire $\hat{X}$ matrix).  We can do this slightly more efficiently if desired (i.e., use additional matrix tricks to compute the entire matrix in one operation), but it's a bit more involved, so we'll just use the "brute force" approach here (it will take a few minutes to run).

```python
Xhat = np.array([predict_item_item(X, W_pearson, movie_means, i) for i in range(X.shape[1])]).T
```

```python
plt.hist(Xhat[0,:], bins=20)
```

{% include image.html img="output_8.svg" %}

One last element to note, though, is that the similarity matrix $W$ was previously an $m \times m$ matrix (number of users by number of users), whereas now it is an $n \times n$ matrix.  Depending on the number of users and items one of these could be much larger than the other, so one should take this into account regarding memory considerations.  Finally, we'll note that for an actual production system, neither of these matrices would be possible to form directly, so instead it's more common to just store some set of nearest neighbors for each user/item, rather than the entire matrix.

## Matrix factorization

Finally, we'll going to last consider a slightly different approach to collaborative filtering, which fits better into the machine learning frameworks we have discussed so far.  For the user-user and item-item approaches, we didn't find the notion of a hypothesis function, loss function, and optimizatin procedure, since there was really no training phase of the algorithm to speak of (the methods _can_ be viewed in this way, with the weighting functions corresponding to different losses, but it's not particularly intuitive).  But with matrix factorization, the correspondence is much more direct, so we'll use this notation here.  The presentation here is somewhat brief, since you'll go through the implemenation and evaluation of matrix factorization in one of the homework questions, but we'll cover the basic ideas here.

The core idea of the matrix factorization approach to matrix factorization is that we're going to approximate the $X$ matrix as a product of two lower-rank matrices

$$
X \approx \hat{X} = UV, \;\; U \in \mathbb{R}^{m \times k}, \; V \in \mathbb{R}^{k \times n}
$$

where $k \ll m,n$.  This is similar to other methods such as PCA that we covered in the course, but crucially, in the collaborative filtering setting we _only_ look to approximate the matrix on terms that are observed in $X$.

Let's define this a bit more formally in terms of the specific elements rows and columns of the matrices $U$ and $V$ respectively.  For each user $i=1,\ldots,m$ we define a set of _user-specific weights_, $u\_i \in \mathbb{R}^k$; and for each item $j$ we define a set of _item-specific weights_, $v\_j \in \mathbb{R}^k$.  Our prediction $\hat{X}\_{ij}$ (which we'll more formally denote as our hypothesis function now), is simply given by

$$
\hat{X}_{ij} \equiv h_\theta(i,j) = u_i^T v_j
$$

where here our parameters are just all the $u$ and $v$ vectors, $\theta = \\{u\_{1:m}, v\_{1:n}\\}$.  One way to interpret this is that you can think of $u\_i$ and $v\_j$ as being something that is _both_ like a feature vector and a parameter vector.  For a given user $i$, our hypothesis function is a linear hypothesis with paramters $u\_i$, and we make our predictions by taking the inner product with these parameters and the item "features" $v\_j$.  Thus, the goal of matrix factorization is to simultaneously learn both the per-user coefficients and the per-item features.

To measure the performance of a given set of parameters, we'll use the squared loss function,

$$
\ell(h_\theta(i,j), X_{ij}) = (h_\theta(i,j) - X_{ij})^2
$$

but evaluatd _only_ for those entries where we have actually observed a rating in $X$.  That is, defining $S$ as the set of observed user/item ratings

$$
S = \{(i,j) : X_{ij} \neq 0\}
$$

our optimization problem is to find parameters that minimize

$$
\DeclareMathOperator*{minimize}{minimize}
\minimize_{u_{1:m},v_{1:n}} \sum_{i,j \in S} (u_i^T v_j - X_{ij})^2.
$$


### Alternating least squares
Finally, how do we solve this optimization problem?  There are actually a number of ways of doing so (including traditional gradient descent), but a common method in practice here is to use a technique called _alternating least squares_.  The method is common enough that it's worth explaining in a bit of detail.  The basic idea here is that it's difficult to solve the above optimization problem globally over both $u$ and $v$ (in fact this is a non-convex problem, so there is the possibility of local optima), if we _fix_ all the $v\_j$ terms then we can find the optimal $u\_i$ terms using simple least-squares; and simiarly, if we fix all the $u\_i$ terms we can solve globally for $v\_j$.  This motivates a simple approach to solving the problem: starting with some initial values (these could even be random), alternatively solve for all the $u\_i$ then all $v\_j$ terms.  These updates are each simple least-squares problems (so they have an anlytical solution), and they can even exploit paralellism to a very high degree (though we won't worry about this here).  Specifically, this algorithm results in the following repeated updates.

$$
\begin{split}
u_i & := \left ( \sum_{j : (i,j) \in S} v_j v_j^T \right )^{-1} \left (\sum_{j : (i,j) \in S} v_j X_ij \right ), \;\; \ i=1,\ldots,m \\
v_j & := \left ( \sum_{i : (i,j) \in S} u_i u_i^T \right )^{-1} \left (\sum_{i : (i,j) \in S} u_i X_ij \right ), \;\; j=1,\ldots,n
\end{split}
$$

where we note that these are just the analytical solutions for least squares applied to the objective above.


### Relation to PCA
As mentioned above, there is a close relationship between matrix factorization for collaborative filtering and PCA.  Both are finding low-rank approximation to some matrix $X$.  But the key difference is that while PCA tries to find an approximation that matches _all_ the entries of $X$ (that is, $S$ would consist of the set of all valid $i,j$ pairs), matrix factorization for collaborative filtering only considers the loss on the observed entries.  Although we won't get into the specifics here, it turns out that this difference means that PCA can be solved optimally an eigenvalue decomposition (or equivalently, a singular value decomposition), whereas matrix factorization cannot be solved in this analytical manner, and the alternating optimization scheme we mentioned above has the potential for local optima.

Because of this, it is somewhat common to initialize matrix factorization with $u$ and $v$ terms determined by PCA (probably subtracting the mean of the data first as in typical PCA, so we don't try too hard _too_ hard to fit the zero entries).  Doing so is not required by any means, but it is a nice way of proving a non-random initial solution to the problem, so that we can begin the matrix factorization steps.
