# Recitation 2

### Homework Tips

* 20 minutes reading documentation will save you 8 hours of writing code
* Be familiar with the [What Can I Ask On Diderot?](www.diderot.one/course/10/dosts/?is_inbox=yes&dost=4658) policy
* Talk to other students taking the course -- they can help you and you can help them.
* Look for the "Common Problems in Homework x" post on Diderot before asking questions online.

### TA Hours

* Come this week! Don't wait until the last week.
* We will only help you with your code if you construct a _minimal counter-example_ : a simplest test case that fails.

```python
import pandas as pd
import sqlite3
import gzip
import scipy.sparse as sp
from random import shuffle
```

## Views and Data

Pandas loads data and presents you with a _view_ of it. Most read-only operations on the `DataFrame` change the view but leave the underlying data intact; this allows you to work quickly (because copying is expensive) with very large datasets (big enough that you can only keep one copy in memory).

Here's an example using a modified version of the [Crimes in Boston](https://www.kaggle.com/AnalyzeBoston/crimes-in-boston) dataset (CC0):

```python
with gzip.open("crime_min.csv.gz", "rt", encoding="UTF-8") as crime_file:
    crime = pd.read_csv(crime_file)
    crime["date"] = pd.to_datetime(crime['date'], infer_datetime_format=True)

with gzip.open("offense_codes.csv.gz", "rt", encoding="UTF-8") as offense_file:
    offense = pd.read_csv(offense_file).drop_duplicates("code")
```

```python
crime
```

<div><small><div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>incident_num</th>
      <th>offense_code</th>
      <th>district</th>
      <th>date</th>
      <th>lat</th>
      <th>lon</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>I182070945</td>
      <td>619</td>
      <td>D14</td>
      <td>2018-09-02 13:00:00</td>
      <td>42.357791</td>
      <td>-71.139371</td>
    </tr>
    <tr>
      <td>1</td>
      <td>I182070943</td>
      <td>1402</td>
      <td>C11</td>
      <td>2018-08-21 00:00:00</td>
      <td>42.306821</td>
      <td>-71.060300</td>
    </tr>
    <tr>
      <td>2</td>
      <td>I182070941</td>
      <td>3410</td>
      <td>D4</td>
      <td>2018-09-03 19:27:00</td>
      <td>42.346589</td>
      <td>-71.072429</td>
    </tr>
    <tr>
      <td>3</td>
      <td>I182070940</td>
      <td>3114</td>
      <td>D4</td>
      <td>2018-09-03 21:16:00</td>
      <td>42.334182</td>
      <td>-71.078664</td>
    </tr>
    <tr>
      <td>4</td>
      <td>I182070938</td>
      <td>3114</td>
      <td>B3</td>
      <td>2018-09-03 21:05:00</td>
      <td>42.275365</td>
      <td>-71.090361</td>
    </tr>
    <tr>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td>319068</td>
      <td>I050310906-00</td>
      <td>3125</td>
      <td>D4</td>
      <td>2016-06-05 17:25:00</td>
      <td>42.336951</td>
      <td>-71.085748</td>
    </tr>
    <tr>
      <td>319069</td>
      <td>I030217815-08</td>
      <td>111</td>
      <td>E18</td>
      <td>2015-07-09 13:38:00</td>
      <td>42.255926</td>
      <td>-71.123172</td>
    </tr>
    <tr>
      <td>319070</td>
      <td>I030217815-08</td>
      <td>3125</td>
      <td>E18</td>
      <td>2015-07-09 13:38:00</td>
      <td>42.255926</td>
      <td>-71.123172</td>
    </tr>
    <tr>
      <td>319071</td>
      <td>I010370257-00</td>
      <td>3125</td>
      <td>E13</td>
      <td>2016-05-31 19:35:00</td>
      <td>42.302333</td>
      <td>-71.111565</td>
    </tr>
    <tr>
      <td>319072</td>
      <td>142052550</td>
      <td>3125</td>
      <td>D4</td>
      <td>2015-06-22 00:12:00</td>
      <td>42.333839</td>
      <td>-71.080290</td>
    </tr>
  </tbody>
</table>
<p>319073 rows × 6 columns</p>
</div></small></div>

It uses a decent amount of memory:

```python
crime.memory_usage(deep=True)
```

```python
str(sum(crime.memory_usage(deep=True)) // 1024 // 1024) + " MB"
```

This doesn't seem like much, but datasets get very large very quickly. Let's select all reports of "GATHERING CAUSING ANNOYANCE":

```python
gca = crime[crime.offense_code == 3302]
gca
```

<div><small><div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>incident_num</th>
      <th>offense_code</th>
      <th>district</th>
      <th>date</th>
      <th>lat</th>
      <th>lon</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>24331</td>
      <td>I182044697</td>
      <td>3302</td>
      <td>D4</td>
      <td>2018-06-09 14:36:00</td>
      <td>42.351251</td>
      <td>-71.073052</td>
    </tr>
    <tr>
      <td>37259</td>
      <td>I182030938</td>
      <td>3302</td>
      <td>D4</td>
      <td>2018-04-25 12:54:00</td>
      <td>42.341318</td>
      <td>-71.078784</td>
    </tr>
    <tr>
      <td>56534</td>
      <td>I182010380</td>
      <td>3302</td>
      <td>A1</td>
      <td>2018-02-08 17:20:00</td>
      <td>42.355407</td>
      <td>-71.063124</td>
    </tr>
    <tr>
      <td>62970</td>
      <td>I182003526</td>
      <td>3302</td>
      <td>E5</td>
      <td>2018-01-13 23:31:00</td>
      <td>42.286228</td>
      <td>-71.124498</td>
    </tr>
    <tr>
      <td>63220</td>
      <td>I182003288</td>
      <td>3302</td>
      <td>B2</td>
      <td>2018-01-12 23:55:00</td>
      <td>42.333220</td>
      <td>-71.109439</td>
    </tr>
    <tr>
      <td>150827</td>
      <td>I172017387</td>
      <td>3302</td>
      <td>A1</td>
      <td>2017-03-04 11:13:00</td>
      <td>42.353110</td>
      <td>-71.064323</td>
    </tr>
    <tr>
      <td>160521</td>
      <td>I172007037</td>
      <td>3302</td>
      <td>A1</td>
      <td>2017-01-26 14:59:00</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>165072</td>
      <td>I172002246</td>
      <td>3302</td>
      <td>A1</td>
      <td>2017-01-09 12:04:00</td>
      <td>42.356502</td>
      <td>-71.062000</td>
    </tr>
    <tr>
      <td>177200</td>
      <td>I162095504</td>
      <td>3302</td>
      <td>C11</td>
      <td>2016-11-22 05:20:00</td>
      <td>42.304815</td>
      <td>-71.072183</td>
    </tr>
    <tr>
      <td>186443</td>
      <td>I162085653</td>
      <td>3302</td>
      <td>A1</td>
      <td>2016-10-19 12:18:00</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>222475</td>
      <td>I162046897</td>
      <td>3302</td>
      <td>B2</td>
      <td>2016-06-13 23:56:00</td>
      <td>42.327541</td>
      <td>-71.099500</td>
    </tr>
    <tr>
      <td>227620</td>
      <td>I162041405</td>
      <td>3302</td>
      <td>B2</td>
      <td>2016-05-27 02:37:00</td>
      <td>42.332590</td>
      <td>-71.100314</td>
    </tr>
    <tr>
      <td>247973</td>
      <td>I162019520</td>
      <td>3302</td>
      <td>C11</td>
      <td>2016-03-13 00:20:00</td>
      <td>42.295072</td>
      <td>-71.047497</td>
    </tr>
    <tr>
      <td>251477</td>
      <td>I162015700</td>
      <td>3302</td>
      <td>B2</td>
      <td>2016-02-27 23:12:00</td>
      <td>42.329751</td>
      <td>-71.098977</td>
    </tr>
    <tr>
      <td>270919</td>
      <td>I152102472</td>
      <td>3302</td>
      <td>B2</td>
      <td>2015-12-12 00:33:00</td>
      <td>42.330570</td>
      <td>-71.099591</td>
    </tr>
    <tr>
      <td>271448</td>
      <td>I152101891</td>
      <td>3302</td>
      <td>B2</td>
      <td>2015-12-09 23:38:00</td>
      <td>42.331286</td>
      <td>-71.102540</td>
    </tr>
    <tr>
      <td>282045</td>
      <td>I152090122</td>
      <td>3302</td>
      <td>B2</td>
      <td>2015-10-30 22:01:00</td>
      <td>42.334278</td>
      <td>-71.102952</td>
    </tr>
    <tr>
      <td>285434</td>
      <td>I152086419</td>
      <td>3302</td>
      <td>C11</td>
      <td>2015-10-17 16:14:00</td>
      <td>42.303565</td>
      <td>-71.078681</td>
    </tr>
    <tr>
      <td>285585</td>
      <td>I152086244</td>
      <td>3302</td>
      <td>B2</td>
      <td>2015-10-17 01:10:00</td>
      <td>42.329645</td>
      <td>-71.097472</td>
    </tr>
    <tr>
      <td>287262</td>
      <td>I152084444</td>
      <td>3302</td>
      <td>B2</td>
      <td>2015-10-11 00:33:00</td>
      <td>42.337683</td>
      <td>-71.096668</td>
    </tr>
    <tr>
      <td>291103</td>
      <td>I152080302</td>
      <td>3302</td>
      <td>B2</td>
      <td>2015-09-26 17:45:00</td>
      <td>42.327016</td>
      <td>-71.105551</td>
    </tr>
    <tr>
      <td>291260</td>
      <td>I152080126</td>
      <td>3302</td>
      <td>B2</td>
      <td>2015-09-26 02:05:00</td>
      <td>42.331513</td>
      <td>-71.104949</td>
    </tr>
    <tr>
      <td>291262</td>
      <td>I152080124</td>
      <td>3302</td>
      <td>B2</td>
      <td>2015-09-26 02:59:00</td>
      <td>42.334097</td>
      <td>-71.102264</td>
    </tr>
    <tr>
      <td>293338</td>
      <td>I152077876</td>
      <td>3302</td>
      <td>B2</td>
      <td>2015-09-19 00:25:00</td>
      <td>42.331348</td>
      <td>-71.103225</td>
    </tr>
    <tr>
      <td>293411</td>
      <td>I152077794</td>
      <td>3302</td>
      <td>E5</td>
      <td>2015-09-18 18:01:00</td>
      <td>42.285370</td>
      <td>-71.172440</td>
    </tr>
    <tr>
      <td>295087</td>
      <td>I152075994</td>
      <td>3302</td>
      <td>B2</td>
      <td>2015-09-13 00:19:00</td>
      <td>42.331836</td>
      <td>-71.104329</td>
    </tr>
    <tr>
      <td>310780</td>
      <td>I152058711</td>
      <td>3302</td>
      <td>C6</td>
      <td>2015-07-16 06:52:00</td>
      <td>42.353208</td>
      <td>-71.046471</td>
    </tr>
  </tbody>
</table>
</div></small></div>

We see that the created frame is a copy. It has a _weak reference_ to the original DataFrame:

```python
gca._is_copy
```

We print out the index and see that it refers to the selected rows in the original DataFrame by row id:

```python
gca.index
```

Making a copy creates a new backing array, independent of the original data:

```python
gca_copy = gca.copy()
print(gca_copy._is_copy)
```

<pre>
None

</pre>


## Databases and Indices

`pandas` and RDBMS software work off of indices. These exist to speed up queries.

The reason indexes are so important is because of [data normalization](https://en.wikipedia.org/wiki/Database_normalization). By allowing us to efficiently connect information dispersed over several tables, we can store data in a way that minimizes redundancy and maximizes throughput. Here's a [basic introduction](https://support.microsoft.com/en-us/help/283878/description-of-the-database-normalization-basics) to the concept.

```python
speed_crime = crime.copy()
speed_crime.index = pd.DatetimeIndex(speed_crime.date)
str(sum(speed_crime.memory_usage(deep=True)) // 1024 // 1024) + " MB"
```

```python
%%timeit
entries = []
fr, to = pd.to_datetime("2016-11-01"), pd.to_datetime("2016-12-01")
for idx, row in speed_crime.iterrows():
    if (idx > fr)  and (idx < to):
        entries.append(row)
```

```python
%%timeit
speed_crime.loc["2016-11-01":"2016-12-01"]
```

<pre>
2.52 ms ± 32.4 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)

</pre>


It's a little quicker! This is why you're expected to complete `hw2_time_series` without looping over the data.

The chief gains in this are from _vectorization_ (which is running as much of the code sequentially as low-level as possible), and then from _indexing_ (which is preparing a lookup data structure to find parts of the data more efficiently).

### Join operation

The use of an index allows us to **join** two tables together using common column values. Lets take a look at the `offense` table:

```python
offense[offense.code.isin([3302, 3116, 1103, 3122, 2401])]
```

<div><small><div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>code</th>
      <th>name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>6</td>
      <td>1103</td>
      <td>CONFIDENCE GAMES</td>
    </tr>
    <tr>
      <td>135</td>
      <td>2401</td>
      <td>AFFRAY</td>
    </tr>
    <tr>
      <td>261</td>
      <td>3116</td>
      <td>HARBOR INCIDENT / VIOLATION</td>
    </tr>
    <tr>
      <td>266</td>
      <td>3122</td>
      <td>AIRCRAFT INCIDENTS</td>
    </tr>
    <tr>
      <td>299</td>
      <td>3302</td>
      <td>GATHERING CAUSING ANNOYANCE</td>
    </tr>
  </tbody>
</table>
</div></small></div>

We've previously set the `code` as the index:

```python
offense_indexed = offense.set_index("code")
offense_indexed
```

<div><small><div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
    </tr>
    <tr>
      <th>code</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1001</td>
      <td>COUNTERFEITING</td>
    </tr>
    <tr>
      <td>1002</td>
      <td>FORGERY OR UTTERING</td>
    </tr>
    <tr>
      <td>1101</td>
      <td>PASSING WORTHLESS CHECK</td>
    </tr>
    <tr>
      <td>1102</td>
      <td>FRAUD - FALSE PRETENSE</td>
    </tr>
    <tr>
      <td>1103</td>
      <td>CONFIDENCE GAMES</td>
    </tr>
    <tr>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td>905</td>
      <td>ARSON - OTHER COMMERCIAL</td>
    </tr>
    <tr>
      <td>906</td>
      <td>ARSON - COMMUNITY/PUB.STRUC.</td>
    </tr>
    <tr>
      <td>907</td>
      <td>ARSON - ALL OTHER STRUCTURES</td>
    </tr>
    <tr>
      <td>920</td>
      <td>ARSON - MOTOR VEHICLES</td>
    </tr>
    <tr>
      <td>930</td>
      <td>ARSON - OTHER</td>
    </tr>
  </tbody>
</table>
<p>425 rows × 1 columns</p>
</div></small></div>

Now we can `join` the two tables, selecting the `offense.name` where `crime.offense_code == offense.code`:

```python
crime.join(offense_indexed, on="offense_code", how="left")
```

<div><small><div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>incident_num</th>
      <th>offense_code</th>
      <th>district</th>
      <th>date</th>
      <th>lat</th>
      <th>lon</th>
      <th>name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>I182070945</td>
      <td>619</td>
      <td>D14</td>
      <td>2018-09-02 13:00:00</td>
      <td>42.357791</td>
      <td>-71.139371</td>
      <td>LARCENY ALL OTHERS</td>
    </tr>
    <tr>
      <td>1</td>
      <td>I182070943</td>
      <td>1402</td>
      <td>C11</td>
      <td>2018-08-21 00:00:00</td>
      <td>42.306821</td>
      <td>-71.060300</td>
      <td>VANDALISM</td>
    </tr>
    <tr>
      <td>2</td>
      <td>I182070941</td>
      <td>3410</td>
      <td>D4</td>
      <td>2018-09-03 19:27:00</td>
      <td>42.346589</td>
      <td>-71.072429</td>
      <td>TOWED MOTOR VEHICLE</td>
    </tr>
    <tr>
      <td>3</td>
      <td>I182070940</td>
      <td>3114</td>
      <td>D4</td>
      <td>2018-09-03 21:16:00</td>
      <td>42.334182</td>
      <td>-71.078664</td>
      <td>INVESTIGATE PROPERTY</td>
    </tr>
    <tr>
      <td>4</td>
      <td>I182070938</td>
      <td>3114</td>
      <td>B3</td>
      <td>2018-09-03 21:05:00</td>
      <td>42.275365</td>
      <td>-71.090361</td>
      <td>INVESTIGATE PROPERTY</td>
    </tr>
    <tr>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td>319068</td>
      <td>I050310906-00</td>
      <td>3125</td>
      <td>D4</td>
      <td>2016-06-05 17:25:00</td>
      <td>42.336951</td>
      <td>-71.085748</td>
      <td>WARRANT ARREST</td>
    </tr>
    <tr>
      <td>319069</td>
      <td>I030217815-08</td>
      <td>111</td>
      <td>E18</td>
      <td>2015-07-09 13:38:00</td>
      <td>42.255926</td>
      <td>-71.123172</td>
      <td>MURDER, NON-NEGLIGIENT MANSLAUGHTER</td>
    </tr>
    <tr>
      <td>319070</td>
      <td>I030217815-08</td>
      <td>3125</td>
      <td>E18</td>
      <td>2015-07-09 13:38:00</td>
      <td>42.255926</td>
      <td>-71.123172</td>
      <td>WARRANT ARREST</td>
    </tr>
    <tr>
      <td>319071</td>
      <td>I010370257-00</td>
      <td>3125</td>
      <td>E13</td>
      <td>2016-05-31 19:35:00</td>
      <td>42.302333</td>
      <td>-71.111565</td>
      <td>WARRANT ARREST</td>
    </tr>
    <tr>
      <td>319072</td>
      <td>142052550</td>
      <td>3125</td>
      <td>D4</td>
      <td>2015-06-22 00:12:00</td>
      <td>42.333839</td>
      <td>-71.080290</td>
      <td>WARRANT ARREST</td>
    </tr>
  </tbody>
</table>
<p>319073 rows × 7 columns</p>
</div></small></div>

## Once More With SQLite

```python
conn = sqlite3.connect(':memory:')
crime.reset_index().to_sql("crime", conn, if_exists="replace")
offense.reset_index().to_sql("offense", conn, if_exists="replace")
conn.execute("PRAGMA automatic_index = false;") # Disable automatic indexing for this demo
conn.commit()
```

```python
conn.execute("SELECT * FROM `crime`;").fetchmany(2)
```

```python
conn.execute("SELECT * FROM `offense`;").fetchmany(2)
```

### SQLite Indexes and Join

We can set SQLite indexes up pretty easily. Lets try the join query from earlier without one first:

```python
join_query = "SELECT * FROM `crime` LEFT JOIN `offense` ON crime.`offense_code` = offense.`code`;"
conn.execute(join_query).fetchmany(2)
```

It works, but how does it happen? We can use the `EXPLAIN QUERY PLAN` instruction to get SQLite to tell us its strategy.

```python
conn.execute(f"EXPLAIN QUERY PLAN {join_query}").fetchall()
```

`SCAN TABLE` means it searches the table by looping over it, the slowest way of doing this. What this means is that SQLite has to loop over the `offense` table once for each entry in `crime`.

Now lets create an index and do this again:

```python
conn.execute(f"CREATE INDEX code_index ON offense(`code`)")
conn.execute(f"EXPLAIN QUERY PLAN {join_query}").fetchall()
```

`SEARCH TABLE` means that the index `code_index` is used to efficiently locate entries inside `offense`. This means that SQLite has to efficiently find entries from `offense` for each entry in `crime`.

## Matrices and Sparse Representation

There's a basic problem with matrices. An $n \times m$ matrix of type `float64` takes $n\times m\times 8$ bytes of memory. That adds up to a lot of space.

$$
\begin{pmatrix}
1 & 2 &  0 &  0 &  0 &  0 \\
 0 & 3 &  0 & 4 &  0 &  0 \\
 0 &  0 & 5 & 6 & 7 &  0 \\
 0 &  0 &  0 &  0 &  0 & 8 \\
\end{pmatrix}$$

That's why we need _sparse representations_.

### Different Representations

There are two different sparse representations we're going to use.

The first is the **Coordinate** or `i, j, v` format. It's called that because we store the matrix as pairs of `i, j` coordinates and `v` values.

```python
i = [0, 1, 2, 3, 4, 5, 6]
j = [6, 1, 4, 3, 2, 0, 5]
v = [1, 2, 3, 4, 5, 6, 7]

m = sp.coo_matrix((v, (i, j)), shape=(7,7))
```

Lets convert the matrix to the dense form to see it:

```python
print(str(m.todense()).replace("0", " "))
```

<pre>
[[            1]
 [  2          ]
 [        3    ]
 [      4      ]
 [    5        ]
 [6            ]
 [          7  ]]

</pre>


We can create matrices that are very large. A $10,000\times 10,000$ matrix of `float64` will take 1.6 GB of memory to create.

```python
size=20000
i = list(range(size)) * 200
j = list(range(size)) * 200
v = list(range(1, size + 1)) * 200
shuffle(j)
m = sp.coo_matrix((v, (i, j)), shape=(size, size))
m
```

```python
str((m.col.nbytes + m.row.nbytes + m.data.nbytes) // 1024 // 1024) + " MB"
```

The most important operation with matrices is matrix multiplication. If we try to use this format to perform matrix multiplication, we'll have to repeatedly search over the entire list. We need a more efficient way to store the sparse values.

That's where the **Compressed Sparse Row** representation comes in:

```python
md = m.tocsr()
md
```

```python
str((md.data.nbytes + md.indptr.nbytes + md.indices.nbytes) // 1024 // 1024) + " MB"
```

The Compressed Sparse Row representation stores each row as a (sometimes sorted) list of `(index, values)` pairs. This allows for much quicker math when the size and density of the matrix is sufficiently large. Here we try it on a matrix of about 1% density:

```python
%%timeit
m * m
```

<pre>
8.97 s ± 25.8 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

</pre>


```python
%%timeit
md * md
```

<pre>
2.76 s ± 6.74 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

</pre>

