# Relational Data

These notes provide a very brief introduction to the relational data and databases.  Chances are, as a data scientist, if you're going to be processing data stored in any more structured form than simple flat files, it's going to be in some kind of database.  And although schemaless/NoSQL/non-relational databases are popular for some applications, in a large number of cases you're going to be dealing with data in a standard relational database (and this format honestly makes sense for the vast majority of use cases that require a database).

While obviously databases are a topic that can't be done any kind of justice in one lecture, these notes will focus on some of the basic ideas of relational databases, and ideally will give you some hints about how to efficiently get data _out_ of a relational database.  Creating and managing such a database, let alone actually coding one, are not topics we'll consider here.

## Overview of relational data

The basic unit in any relational data is the notion of a "relation", but since no one uses that term anymore, just note that this actually is interchangeable with the common definition we have of "table" (though a relation has the additional constraints the rows in the table cannot be duplicates).  To make this concerete, let's consider a "Person" table for people involved with this class.

<div class="center-text" markdown="1">
**Person**
</div>

| ID | Last Name | First Name | Role |
| :---: | :---: | :---: | :---: |
| 1 | Kolter | Zico | Instructor |
| 2 | Xi | Edgar | TA |
| 3 | Lee | Mark | TA |
| 4 | Mani | Shouvik | TA |
| 5 | Gates | Bill | Student |
| 6 | Musk | Elon | Student |

This kind of data representation is so familiar to us that it hardly necessitates any explanation, but a few points of terminology in relational database speak are useful.  In our context, rows are called _tuples_ or _records_, and they represent a single instance of this relation; as mentioned above, the entirety of the row must also be unique (i.e., there cannot be two rows that have identical entries in all columns).  This uniqueness requirement can be trivially satisfied by having a unique ID identifier in each tuple, and we'll discuss this more below.  Columns in the relation are called _attributes_, and specify some feature contained in each of the tuples.

### Multiple relations

Of course, a single table isn't particular interesting as a rich data source; we could have done (and in fact do) store the exact same information in a CSV file.  Where relational data becomes interesting is when we have multiple tables and explicit relationships between them.  For example, considering the case above, the "Role" attribute in our Person table is not particularly well defined.  What is the data type of this attribute?  A string? If so, we could enter anything we want here, which would not be ideal for maintaining a large collection of data, say if each student has to remember to type their role exactly as "Student" (with capitalization intact, of course).  A better alternative is to create a separate "Role" table, that lists the allowable roles for the course:

<div class="center-text" markdown="1">
**Role**
</div>

| ID | Name |
| :---: | :---: |
| 1 | Instructor |
| 2 | TA |
| 3 | Student |

Using this table, we can replace the "Role" attribute in our Person table with a "Role ID" attribute that points to the ID of the respective role for each person:

<div class="center-text" markdown="1">
**Person**
</div>

| ID | Last Name | First Name | Role ID |
| :---: | :---: | :---: | :---: |
| 1 | Kolter | Zico | 1 |
| 2 | Xi | Edgar | 2 |
| 3 | Lee | Mark | 2 |
| 4 | Mani | Shouvik | 2 |
| 5 | Gates | Bill | 3 |
| 6 | Musk | Elon | 3 |

### Primary and foreign keys

Of course, we haven't really gained _that_ much benfit here, since it's still possible for the "Role ID" attribute to contain some number, e.g. 4, that doesn't have a corresponding entry in the Role table.  This brings us to the concepts of _keys_ and _constraints_.

In the above examples, "ID" attribute serves as what is called a _primary key_.  A primary key is a _unique_ identifier for each row in the table.  It is common to have a single column (like we do here as the "ID" column) serve as the primary key, but that is not required; the primary key can consist of multiple columns so long as they are unique in every row.  Every relation (table) in the database must have exactly one primary key.

A _foreign key_ is an attribute that "points" to the primary key of another table.  Thus, in the above example, the "Role ID" attribute in the Person table is a foreign key, pointing to the primary key of the Role table.  And the _foreign key constraint_ enforces the fact that the foreign key must point to a valid primary key in the relevant table.  Note, of course, that we _could_ have simply had a single-row Role table with the name itself being the primary key, and had a similar foreign key constraint, but using the ID as a primary key is much more typical.

The foreign key constraint help enforce consistency of the database, and also forces us to be careful when we delete elements.  For example, if we delete a row from Role, we must also delete all the rows from Person that point to that primary key, or the foreign key constraint would be violated.

### Indexes (not indices)

Finally, the last "basic" element we'll consider in a relational database is the notion of an index.  Indexes are created to "quickly" look up rows by some of their attributes.  For example, suppose we wanted to find all the people in our Person table with the last name of "Gates".  Naively, there would be no way to do this except search over the entire table to see every column that matched this last name.  Instead of doing this, we can build an index on the "Last Name" attribute to provide an efficient means for retreiving tuples based upon last name.  To see how this works conceptually, consider a slightly more explicit form of the Person table, where we explicitly denote the location, on disk or in memory, where each tuple occurs (preusming here that each row takes exactly 100 bytes):

<div class="center-text" markdown="1">
**Person**
</div>

| Location | ID | Last Name | First Name | Role ID |
| :---: | :---: | :---: | :---: | :---: |
| **0** | 1 | Kolter | Zico | 1 |
| **100** | 2 | Xi | Edgar | 2 |
| **200** | 3 | Lee | Mark | 2 |
| **300** | 4 | Mani | Shouvik | 2 |
| **400** | 5 | Gates | Bill | 3 |
| **500** | 6 | Musk | Elon | 3 |

You can think of an index like a table with just the indexed attribute and the location field (location in the original table), but _sorted_ by the indexed attribute.  So for instance, an index on the Last Name attribute would take the form:

|Last Name| Location |
| :---: | :---: |
| Gates | **400** |
| Kolter | **0** |
| Lee | **200** |
| Mani | **300** |
| Musk | **500** |
| Xi | **100** |

Because this list is sorted, we can look up any element in $O(\log n)$ time, and then find its corresponding location in the original table.  We can also perform things such as range queries, quickly finding all the entries where the last name starts with "M" (search for the first one, then sequentially scan until the condition is no longer met).  Internally, indexes are frequently implemented with B-trees, which are a type of search tree that in made to be particularly efficient on disk.

As a final note, the primary key associated with a table will always have an index associated with it, so by default it will also provide a fast way to look up rows in the table.  It is also possible to have indexes over multiple columns (with some ordering), by just sorting over both columns.

## Entity relationships

The nature of inter-table relationships via primary and foreign keys actually leads to a number of different possible entity relationships (i.e., relationships between a row in one table and a row in another).  Some of the common types are:
- One-to-one
- One-to-zero/one
- One-to-many
- Many-to-many

We'll cover each of these in turn, though perhaps not in the order you'd imagine.

### One-to-many
Instead of beginning with the simpler-seeming one-to-one relation, let's begin with the example we have already actually used in the above setting: the one-to-many relationship.  In the above table example, the Role - Person tables had a one-to-many relationship: one role can have many different people associated with it, but each person must have exactly one role.  This relationship is illustrated with the following notation:

{% include image.html img="one_to_many.svg" caption="One-to-many relationship."%}

The one-to-many relationship is represented in the database using the foreign key constraint discussed earlier.  However, it's also important to note a subtle but important difference between the underlying relationship (which is an inherrent relationship between the two type entities that the tables represent), and the way it is actually expressed within the relational database.  This distinction should become more apparent when we discuss many-to-many relationships.

### One-to-one (and one-to-zero/one)

The next relationship we'll conider is that of one-to-one relationships.  As you may expect, this occurs when there exist two tables such that there every entry of the table has exactly one entry in the corresponding table, and vice versa.  Let's consider a simple example, of an corresponding "Andrew ID" table that could associated with our Person table

<div class="center-text" markdown="1">
**Andrew ID**
</div>

| Person ID | Andrew ID |
| :---: | :---: |
| 1 | zkoler |
| 2 | esx |
| 3 | marklee |
| 4 | shouvikm |
| 5 | bgates |
| 6 | muskleman |

In this instance, every person has exactly one Andrew ID, and every Andrew ID is associated with exactly one person. As you might imagine, the relationship is denoted as follows.

{% include image.html img="one_to_one.svg" caption="One-to-one relationship."%}

Despite its simplicity, this type of relation is actually not used that much in practice for an obvious reason: if every entry in each table has a corresponding entry in the other, we could simplify the database design by simply combining these into the same table with attributes from both.  I.e., in the case above we could simply add an additional "Andrew ID" attribute to the original Person table, and accomplish the same thing as with two tables and the one-to-one relationship, with less complexity to the database.  When you do see one-to-one relationships in databases, it is often because some tables need to be (or simply are in practice) exact mirrors from tables in other system, so a database designer may not want to merge two tables with all corresponding elements.

What is more common in practice is a one-to-zero/one relationship, where one table always has a corresponding entry in the other, but the inverse relationship is optional.  Let's take an example of a "Grades" table, that will store grades for the students in our class.  It may look something like this:

<div class="center-text" markdown="1">
**Grades**
</div>

| Person ID | HW1 Grade | HW2 Grade |
| :---: | :---: | :---: |
| 5 | 85 | 95 |
| 6 | 80 | 60 |

Every entry in the grades table must refer to an entry in the Person table, but not every Person has an entry in the grades table, i.e., the Person and Grades entities have one-to-zero/one relationship.  This is because, depending on one's role in the class, it may or may not make sense to have grades: instructors and TAs won't have grades, just students.  Unlike in the pure one-to-one relationship, in this case it does seem reasonable to have two separate tables for the grades and people, because it seems uncecessary for instance to simply include these attributes in the Person table when they aren't actually relevant to for all people.  The one-to-zero/one relationship is denoted like this:

{% include image.html img="one_to_zero_one.svg" caption="One-to-zero/one relationship."%}

There are actually a lot of these decorations that you'll see if you look at database schema diagrams.  In the case above, the bar denotes that it's entries in that table are required while the circle denotes that the entries are options.  But honestly, for most relationship types I'll need to consult a reference to remember what all the different annotations of the relationship symbols mean, beyond the basic one-to-many notation, etc.

### Many-to-many

We'll end by discussing the most involved type of relationship, the many-to-many relationship, where each row in a one table can relate to many rows in another, and vice versa.  As an example, let's consider an alternative way to represent student grades than what we considered above.  A problem with the above representation is that, if we have a separate "Homework" table (that describes each assignment, for instance, in terms of the total points, release date, due date, etc), then there is no way to capture the fact that the "HW1 Grade" attribute in the Grades tably actually pointed to a particular row in the homework table; this would simply have to be implicit knowledge for a user of the database.  But we can actually do something that is a bit more principled than this.  First, let's consider an instantion of the Homework table we just described:

<div class="center-text" markdown="1">
**Homework**
</div>

| ID | Name | Points | Release Date | Due Date |
| :---: | :---: | :---: | :---: | :---: |
| 1 | HW1 | 100 | 2018-01-24 | 2018-02-07 |
| 2 | HW2 | 100 | 2018-02-07 | 2018-02-21 |

Now we'd like to create an additional table that specifies, for each relevant student and each relevant homework, what the student's grade was.  We can do this via an _associative table_, which links together entries in the Person and Homework tables, and additional assigns a score to each relevant pairing.  For example, the same grades as we highlighted in the past example, could be represented by the table

<div class="center-text" markdown="1">
**Person Homework**
</div>

| Person ID | Homework ID | Score |
| :---: | :---: | :---: |
| 5 | 1 | 85 |
| 5 | 2 | 95 |
| 6 | 1 | 80 |
| 6 | 2 | 60 |

This associative table is a bit interesting.  Unlike previous tables (that all have an individual column as their primary key, usually just the ID column), the primary key of this table is actually the (Person ID, Homework ID) combination, each of which are also foreign keys to the other respective table.  Note that (Person ID, Homework ID) needs to be the primary key here, because we need to require that each person/homework combination has only _one_ score in the table, not potentially multiple scores.

At a conceptual level, though, what this tables allows is a many-to-many relationship between the homework and person entities: each person can complete multiple homeworks, and each homework can be completed by multiple people.  We denote this relationship as:

{% include image.html img="many_to_many.svg" caption="Many-to-many relationship."%}

Note that from a perspective of the tables themselves, it may also seems reasonable to write the relationship via two one-to-many relationships, explicitly encoding the associative table:

{% include image.html img="many_to_many2.svg" caption="An alternative possibility for the many-to-many relationship."%}

But this is not really the correct relationship from a philosophical sense.  Remember that relationships in this setting are reall relationships between the underlying entities that the database representes, not between the actual tables in the database.  In this case, the "real" underlying relationship is a many-to-many relationship between people and homeworks, and the way that it is enabled via an associative table is simply an "implementation detail."  Thus, if our goal is to describe the actual underlying relationships with entities, we should use the many-to-many relationship here, and realize that we are going to use the associate table as a means of implementing this relationship.

## Libraries for relational data

There are a number of Python libraries that handle relational data, typically written as interfaces to several differen relational database management systems (software such as PostgreSQL, MySQL, or a variety of others).  [**Note**: one aside is that software like PostreSQL and MySQL is most correctly refered to as a relation database managment system (RDBMS), _not_ a database.  The database is the actual tables and records specifying an actual collection of data.]

In this class, though, we'll mainly interact with relational data through two libraries: [Pandas](https://pandas.pydata.org/) and [SQLite](https://www.sqlite.org/).  These are especially simple libraries as far as real databases go: Pandas is decidedly _not_ a real relational database system (although it provides functions that mirror some functionality of them), whereas SQLite _is_ a "real" RDBMS, but an extremely simple one without the standard client/server architecture of virtually any real production database.  Nonetheless, for many data science problems they will suffice, and so we focus on them here.

### Pandas
We have already briefly seen Panda when we discussed data collection, and it ends up being one of the most useful Python libraries for data science.  As we mentioned above (but we're going to repeat this fact many times), Pandas is not a relational database library, but instead a "data frame" library.  You can think of a data frame as being essentially like a 2D array, except that entires in the data frame can be any type of Python object (and have mixed types within the array), and the rows/columns can have "labels" instead of just integer indices like in a standard array.

Let's see how to first create a data frame in Pandas that mirrors our Person table above (we'll leave out the "Role ID" column just to keep things simple).

```python
import pandas as pd

df = pd.DataFrame([(1, 'Kolter', 'Zico'), 
                   (2, 'Xi', 'Edgar'),
                   (3, 'Lee', 'Mark'), 
                   (4, 'Mani', 'Shouvik'),
                   (5, 'Gates', 'Bill'),
                   (6, 'Musk', 'Elon')], 
                  columns=["id", "last_name", "first_name"])
df
```

<div><small><div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>last_name</th>
      <th>first_name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Kolter</td>
      <td>Zico</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Xi</td>
      <td>Edgar</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Lee</td>
      <td>Mark</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>Mani</td>
      <td>Shouvik</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>Gates</td>
      <td>Bill</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>Musk</td>
      <td>Elon</td>
    </tr>
  </tbody>
</table>
</div></small></div>

The first column in the displayed data frame is the "index" of each row.  Unfortunately the terminology here clashes with database terminology, but "index" for Pandas actually means something more like "primary key" in a database table (though with the exception that it _is_ possible to have duplicate entries, though I'd always recommend avoiding this in Pandas).  That is, an index (if done right, without duplicate indices) is a identifier for each row in the database.  We can set the index to one of the existing columns using the `.set_index()` call.

```python
df.set_index("id")
```

<div><small><div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>last_name</th>
      <th>first_name</th>
    </tr>
    <tr>
      <th>id</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>Kolter</td>
      <td>Zico</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Xi</td>
      <td>Edgar</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Lee</td>
      <td>Mark</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Mani</td>
      <td>Shouvik</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Gates</td>
      <td>Bill</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Musk</td>
      <td>Elon</td>
    </tr>
  </tbody>
</table>
</div></small></div>

But you need to be very careful about one thing here.  By default, most Pandas operations, like `.set_index()` and many others, and _not_ done in place.  That is, while the `df.set_index("id")` call above _returns_ a copy of the `df` dataframe with the index set to the `id` column (remember that Jupyter notebook displays the return value of the last line in a cell), the original `df` object is actually unchanged here:

```python
df
```

<div><small><div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>last_name</th>
      <th>first_name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Kolter</td>
      <td>Zico</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Xi</td>
      <td>Edgar</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Lee</td>
      <td>Mark</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>Mani</td>
      <td>Shouvik</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>Gates</td>
      <td>Bill</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>Musk</td>
      <td>Elon</td>
    </tr>
  </tbody>
</table>
</div></small></div>

If we want to actually change the `df` object itself, you need to use the `inplace=True` flag for these functions (or assign the original object to the result of a function, but this isn't as clean):

```python
df.set_index("id", inplace=True)
df
```

<div><small><div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>last_name</th>
      <th>first_name</th>
    </tr>
    <tr>
      <th>id</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>Kolter</td>
      <td>Zico</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Xi</td>
      <td>Edgar</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Lee</td>
      <td>Mark</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Mani</td>
      <td>Shouvik</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Gates</td>
      <td>Bill</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Musk</td>
      <td>Elon</td>
    </tr>
  </tbody>
</table>
</div></small></div>

There are also some peculiarities that you'll get used to with Pandas, such as the fact that if you select a single column or row from a Pandas DataFrame, you don't get a single-column or single-row DataFrame, but an alternative Pandas type called a `Series`, which is kind of like a 1D data frame (but it has some different indexing rules).  There are reasons behind all these design decisions, even if they can seem a bit arbitrary at first.

We won't go into too much detail about Pandas here, but you should familiarize yourself with some of the basic strategies for accessing elements in a Pandas DataFrame.  There are actually a few different ways to do this, but I usually advocate for using the `.loc` or `.iloc` properties when you want to access or set individual elements (other strategies often lead to confusion when accessing by different types of indices, etc.

### Common Pandas data access

Let's consider a few of the common ways to access or set data in a Pandas DataFrame.  You can access individual elements using the `.loc[row, column]` notation, where `row` denotes the index you are searching for and `column` denotes the column name.  For example, to access the last name of person with ID 1 we would execute:

```python
df.loc[1, "last_name"]
```

If we want to access _all_ last names, (or all elements in a particular row), we use the `:` wildcard.  For example

```python
df.loc[:, "last_name"]
```

Notice the output here looks a bit different than the nicely-formed typical Pandas output, which stems from the fact (mentioned previously), that this returne object is actually not a Pandas DataFrame, but a Pandas Series, which (apparently) no one wants to write a nice display routine for.  If we do want to get a DataFrame with just this one column, we can get a "list" of columns (with just one elements)

```python
df.loc[:, ["last_name"]]
```

<div><small><div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>last_name</th>
    </tr>
    <tr>
      <th>id</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>Kolter</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Xi</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Lee</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Mani</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Gates</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Musk</td>
    </tr>
  </tbody>
</table>
</div></small></div>

which now gives us the desired 1D data frame.  We can do a similar thing with row indexes.

```python
df.loc[[1,2],:]
```

<div><small><div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>last_name</th>
      <th>first_name</th>
    </tr>
    <tr>
      <th>id</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>Kolter</td>
      <td>Zico</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Xi</td>
      <td>Edgar</td>
    </tr>
  </tbody>
</table>
</div></small></div>

which will select only a subject of the allowable rows.

We can additionally use `.loc` to change the content of existing entries:

```python
df.loc[1,"last_name"] = "Kilter"
df
```

<div><small><div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>last_name</th>
      <th>first_name</th>
    </tr>
    <tr>
      <th>id</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>Kilter</td>
      <td>Zico</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Xi</td>
      <td>Edgar</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Lee</td>
      <td>Mark</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Mani</td>
      <td>Shouvik</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Gates</td>
      <td>Bill</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Musk</td>
      <td>Elon</td>
    </tr>
  </tbody>
</table>
</div></small></div>

We can even add additional rows/columns that don't exist.

```python
df.loc[7,:] = ('Moore', 'Andrew')
df
```

<div><small><div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>last_name</th>
      <th>first_name</th>
    </tr>
    <tr>
      <th>id</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>Kilter</td>
      <td>Zico</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Xi</td>
      <td>Edgar</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Lee</td>
      <td>Mark</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Mani</td>
      <td>Shouvik</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Gates</td>
      <td>Bill</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Musk</td>
      <td>Elon</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Moore</td>
      <td>Andrew</td>
    </tr>
  </tbody>
</table>
</div></small></div>

Finally, remember that `.loc` always indexes based upon the "index" (i.e., effectively primary key) of the data frame along with the column name.  If you want to instead access based upon positional index (i.e., using 0-indexed counters for both the rows and columns), you can use the `.iloc` property

```python
df.iloc[4,1]
```

### SQLite

Unlike Pandas, SQLite actually is a full-featured database, but unlike most production databases, it does not use a client/server model.  Databases are instead stored direclty on disk and accessed just via the library.  This has the advantage of being very simple, with no server to configure and run, but for large applications it is typically insufficient: because files are not very good at concurrent access (that is, many different processes/threads cannot simultaneously read and write from a single file), the system is not ideal for very large databases where multiple threads need to be constant readings from and writing to the database.  Note that SQLite _does_ have some limited forms of concurrency in this respect, but nothing sophisticated when compared to larger scale databases.  If you do want to investigate a more "production strength" client/server database, I'd highly recommend looking into PostreSQL.

SQLite, as the name suggests, uses the SQL (structured query language) language for interacting with the database; note both "Sequel" and "Ess Queue Ell" are acceptable prononciations of SQL, but I personally learned it as "Sequel", so would be completely unable to do anything else.

Interacting with SQLite (or any other SQL-based database) from Python is not ideal, because you typically use Python code to generate SQL expressions as strings, then execute them, which is not the most beautiful coding paradigm.  For simple databases, though, it usually suffices to get the job done.

Let's look at how to create a simple database with the "Person" and "Grades" tables that we had considered earlier.

```python
import sqlite3
conn = sqlite3.connect("database.db")
cursor = conn.cursor()

### when you are done, call conn.close()
```

This code imports the library, creates a connection to the "database.db" file (it will create it if it does not already exist), and then creates a "cursor" into the database.  The notion of cursor is common to a lot of database libraries, but essentially a cursor is an object that allows us to interact with the database.  If we want to create the Person and Grades tables we saw above (to keep things simple, and later to illustrate joins, we'll use the first version of the Grages table, with no associative table), we would use the following syntax.

```python
cursor.execute("""
CREATE TABLE person (
    id INTEGER PRIMARY KEY,
    last_name TEXT,
    first_name TEXT
);""")

cursor.execute("""
CREATE TABLE grades (
    person_id INTEGER PRIMARY KEY,
    hw1_grade INTEGER,
    hw2_grade INTEGER
);""")

conn.commit()
```

Let's insert some data into these tables.  The syntax for this operation, hopefully fairly straightforward is given by the following:

```python
cursor.execute("INSERT INTO person VALUES (1, 'Kolter', 'Zico');")
cursor.execute("INSERT INTO person VALUES (2, 'Xi', 'Edgar');")
cursor.execute("INSERT INTO person VALUES (3, 'Lee', 'Mark');")
cursor.execute("INSERT INTO person VALUES (4, 'Mani', 'Shouvik');")
cursor.execute("INSERT INTO person VALUES (5, 'Gates', 'Bill');")
cursor.execute("INSERT INTO person VALUES (6, 'Musk', 'Elon');")

cursor.execute("INSERT INTO grades VALUES (5, 85, 95);")
cursor.execute("INSERT INTO grades VALUES (6, 80, 60);")
cursor.execute("INSERT INTO grades VALUES (100, 100, 100);")
```

[Note that I'm an additional row to the `grades` tables with `person_id=100` for illustration purposes later when we talk about joins.  Notice that this actually would violate the consistency of the data if we had required `grades.person_id` to be a true foreign key, but since we didn't bother, it works fine].  If we want to see what has been added to the database, we can do this the "SQLite Python" way, which involves running a query and then iterating over the rows in a result returned by a `cursor.execute()` result, as so:

```python
for row in cursor.execute("SELECT * FROM person;"):
    print(row)
```

<pre>
(1, &#x27;Kolter&#x27;, &#x27;Zico&#x27;)
(2, &#x27;Xi&#x27;, &#x27;Edgar&#x27;)
(3, &#x27;Lee&#x27;, &#x27;Mark&#x27;)
(4, &#x27;Mani&#x27;, &#x27;Shouvik&#x27;)
(5, &#x27;Gates&#x27;, &#x27;Bill&#x27;)
(6, &#x27;Musk&#x27;, &#x27;Elon&#x27;)

</pre>


Alternatively, it can be handy to dump the results of a query directly into a Pandas DataFrame.  Fortunately, Pandas provides a nice call for doing this, the `pd.read_sql_query()` function, with takes the database connection and an optional argument to set the index of the Pandas dataframe to be one of the columns.

```python
pd.read_sql_query("SELECT * from person;", conn, index_col="id")
```

<div><small><div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>last_name</th>
      <th>first_name</th>
    </tr>
    <tr>
      <th>id</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>Kolter</td>
      <td>Zico</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Xi</td>
      <td>Edgar</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Lee</td>
      <td>Mark</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Mani</td>
      <td>Shouvik</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Gates</td>
      <td>Bill</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Musk</td>
      <td>Elon</td>
    </tr>
  </tbody>
</table>
</div></small></div>

The `SELECT` statement is probably the SQL command you'll use most in data science: it is used to query data from th database.  We won't here go into a full description of the `SELECT` statement, except to say that the most common syntax is something like:

`<columns>` here will be a comma-separated list of all the columns to select, or the wildcard `*` to denote all columns.  `<tables>` is a comma-separated list of tables.  And `<conditions>` is a list of conditions, typically separated by `AND` if there are multiple conditions, that specify what subset to return.  For example, let's see how to select all the last name (and id) from the persons table with `id > 2`

```python
pd.read_sql_query("SELECT id,last_name FROM person WHERE id > 2;", conn, index_col="id")
```

<div><small><div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>last_name</th>
    </tr>
    <tr>
      <th>id</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3</th>
      <td>Lee</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Mani</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Gates</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Musk</td>
    </tr>
  </tbody>
</table>
</div></small></div>

**Note**: SQL commands are case insensitive, so using all caps for the statements is purely a convention.

Lastly, we can also delete values from tables using the `DELETE FROM` SQL command, using a similar `WHERE` clause as in the `SELECT` command.

```python
cursor.execute("INSERT INTO person VALUES (7, 'Moore', 'Andrew');")
pd.read_sql_query("SELECT * from person;", conn, index_col="id")
```

<div><small><div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>last_name</th>
      <th>first_name</th>
    </tr>
    <tr>
      <th>id</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>Kolter</td>
      <td>Zico</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Xi</td>
      <td>Edgar</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Lee</td>
      <td>Mark</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Mani</td>
      <td>Shouvik</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Gates</td>
      <td>Bill</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Musk</td>
      <td>Elon</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Moore</td>
      <td>Andrew</td>
    </tr>
  </tbody>
</table>
</div></small></div>

```python
cursor.execute("DELETE FROM person where id = 7;")
pd.read_sql_query("SELECT * from person;", conn, index_col="id")            
```

<div><small><div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>last_name</th>
      <th>first_name</th>
    </tr>
    <tr>
      <th>id</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>Kolter</td>
      <td>Zico</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Xi</td>
      <td>Edgar</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Lee</td>
      <td>Mark</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Mani</td>
      <td>Shouvik</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Gates</td>
      <td>Bill</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Musk</td>
      <td>Elon</td>
    </tr>
  </tbody>
</table>
</div></small></div>

## Joins

The last topic we'll consider here are joins between tables.  Briefly, join operations multiple multiple tables into a single relation, matching between attributes in the two tables.  There are four types of joins, though only the first two are used much in practice:

1. Inner
2. Left
3. Right
4. Outer

You join two tables _on columns_ from each table, where these columns specify how to match the rows between the two columns.  This should become more clear with a few examples.  In the examples that follow, we're going to consider our Person and Grades tables, that we just created above, and we're join to join the tables on the `person.id` and `grades.person_id` columns.

### Inner joins

If you don't know what type of join you want, you probably want an inner join.  This does the "obvious" thing, of only returning those rows where the two columns in each table have matching values, and it appends the rows together for each of these matching rows.  In our case, an inner join between the Person and Grades table would return the following:

| ID | Last Name | First Name | HW1 Grade | HW2 Grade |
| :---: | :---: | :---: | :---: | :---: |
| 5 | Gates | Bill | 85 | 95 |
| 6 | Musk | Elon | 80 | 60 |

Let's see how this can be done programatically, first in Pandas and then in SQL.  In Pandas, you should do joins with the `.merge()` command: there is an alternative `.join()` command, but this always assumes you want to join on the index column for one of the data frames, and _not_ the index frame for another, and overall is just a special case of `.merge()` so I prefer to learn the generic call.

```python
df_person = pd.read_sql_query("SELECT * FROM person", conn)
df_grades = pd.read_sql_query("SELECT * FROM grades", conn)
df_person.merge(df_grades, how="inner", left_on = "id", right_on="person_id")
```

<div><small><div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>last_name</th>
      <th>first_name</th>
      <th>person_id</th>
      <th>hw1_grade</th>
      <th>hw2_grade</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5</td>
      <td>Gates</td>
      <td>Bill</td>
      <td>5</td>
      <td>85</td>
      <td>95</td>
    </tr>
    <tr>
      <th>1</th>
      <td>6</td>
      <td>Musk</td>
      <td>Elon</td>
      <td>6</td>
      <td>80</td>
      <td>60</td>
    </tr>
  </tbody>
</table>
</div></small></div>

Hopefully the syntax of `.merge()` looks straightforward: you call the method on the "left" data frame, and pass the "right" data frame as the first argument.  The `how` parameter specifies the type of join (here inner), and the `left_on` and `right_on` arguments specify the column name that you want to join on for the left and right data frames respectively.  If you alternatively want to join on the index for the left or right data frame, you specify the `left_index` or `right_index` parameters as so:

```python
df_person.set_index("id", inplace=True)
df_grades.set_index("person_id", inplace=True)
```

```python
df_person.merge(df_grades, how="inner", left_index=True, right_index=True)
```

<div><small><div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>last_name</th>
      <th>first_name</th>
      <th>hw1_grade</th>
      <th>hw2_grade</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5</th>
      <td>Gates</td>
      <td>Bill</td>
      <td>85</td>
      <td>95</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Musk</td>
      <td>Elon</td>
      <td>80</td>
      <td>60</td>
    </tr>
  </tbody>
</table>
</div></small></div>

This call is maybe a bit cleaner, as it doesn't include the (duplicate) columns you match on as in the previous example, but this is a minor difference.

In SQL, an inner join is specified by the `WHERE` clause in a `SELECT` statement.

```python
pd.read_sql_query("SELECT * FROM person, grades WHERE person.id = grades.person_id" , conn)
```

<div><small><div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>last_name</th>
      <th>first_name</th>
      <th>person_id</th>
      <th>hw1_grade</th>
      <th>hw2_grade</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5</td>
      <td>Gates</td>
      <td>Bill</td>
      <td>5</td>
      <td>85</td>
      <td>95</td>
    </tr>
    <tr>
      <th>1</th>
      <td>6</td>
      <td>Musk</td>
      <td>Elon</td>
      <td>6</td>
      <td>80</td>
      <td>60</td>
    </tr>
  </tbody>
</table>
</div></small></div>

Exactly the same behavior as our Pandas join, but the advantage of course is that for very complex joins and large database queries, a true database will be faster at performing queries like this than the Pandas code.

### Left joins

Whereas an inner join only kept those rows with corresponding entires in both tables, a left join will keep _all_ the items in the left table, and add in the attribution from the right table (filling with NaNs if no match exists in the right table).  Any row value that occurs in the right table but not the left table is discarded.

For the rest of this section, we'll simply write the Pandas code to perform the relevant join, then show the output it produces, rather than explicitly write the table that results.

```python
df_person = pd.read_sql_query("SELECT * FROM person", conn)
df_grades = pd.read_sql_query("SELECT * FROM grades", conn)
df_person.merge(df_grades, how="left", left_on = "id", right_on="person_id")
```

<div><small><div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>last_name</th>
      <th>first_name</th>
      <th>person_id</th>
      <th>hw1_grade</th>
      <th>hw2_grade</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Kolter</td>
      <td>Zico</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Xi</td>
      <td>Edgar</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Lee</td>
      <td>Mark</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>Mani</td>
      <td>Shouvik</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>Gates</td>
      <td>Bill</td>
      <td>5.0</td>
      <td>85.0</td>
      <td>95.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>Musk</td>
      <td>Elon</td>
      <td>6.0</td>
      <td>80.0</td>
      <td>60.0</td>
    </tr>
  </tbody>
</table>
</div></small></div>

Note for the two students, we properly fill in the grades, but for everyone else will fill in NaN values for the grades, because the person did not have any associated grades in the table.

SQL syntax for left join use the `LEFT JOIN` statement, as follows:

```python
pd.read_sql_query("SELECT * FROM person LEFT JOIN grades ON person.id = grades.person_id" , conn)
```

<div><small><div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>last_name</th>
      <th>first_name</th>
      <th>person_id</th>
      <th>hw1_grade</th>
      <th>hw2_grade</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Kolter</td>
      <td>Zico</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Xi</td>
      <td>Edgar</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Lee</td>
      <td>Mark</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>Mani</td>
      <td>Shouvik</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>Gates</td>
      <td>Bill</td>
      <td>5.0</td>
      <td>85.0</td>
      <td>95.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>Musk</td>
      <td>Elon</td>
      <td>6.0</td>
      <td>80.0</td>
      <td>60.0</td>
    </tr>
  </tbody>
</table>
</div></small></div>

### Right joins

A right join does what you might expect, the converse of the left join, where all the rows in the right matrix are kept.  While SQLite has no syntax for right joins (you can achieve the same results by simply reversing the order of the two tables and doing a left join), Pandas does have built-in syntax for the right join

```python
df_person.merge(df_grades, how="right", left_on = "id", right_on="person_id")
```

<div><small><div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>last_name</th>
      <th>first_name</th>
      <th>person_id</th>
      <th>hw1_grade</th>
      <th>hw2_grade</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5.0</td>
      <td>Gates</td>
      <td>Bill</td>
      <td>5</td>
      <td>85</td>
      <td>95</td>
    </tr>
    <tr>
      <th>1</th>
      <td>6.0</td>
      <td>Musk</td>
      <td>Elon</td>
      <td>6</td>
      <td>80</td>
      <td>60</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
    </tr>
  </tbody>
</table>
</div></small></div>

### Outer joins

Finally, outer joins (also called a cross product) do what you may expect, and keep all rows that occur in either table, so essentially take the union of the left and right joins.  Again, SQLite has no syntax for it (you can achieve the same thing via a `UNION` statement, but we won't cover it), but Pandas has the function.

```python
df_person.merge(df_grades, how="outer", left_on = "id", right_on="person_id")
```

<div><small><div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>last_name</th>
      <th>first_name</th>
      <th>person_id</th>
      <th>hw1_grade</th>
      <th>hw2_grade</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>Kolter</td>
      <td>Zico</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2.0</td>
      <td>Xi</td>
      <td>Edgar</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3.0</td>
      <td>Lee</td>
      <td>Mark</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.0</td>
      <td>Mani</td>
      <td>Shouvik</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5.0</td>
      <td>Gates</td>
      <td>Bill</td>
      <td>5.0</td>
      <td>85.0</td>
      <td>95.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6.0</td>
      <td>Musk</td>
      <td>Elon</td>
      <td>6.0</td>
      <td>80.0</td>
      <td>60.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>100.0</td>
      <td>100.0</td>
      <td>100.0</td>
    </tr>
  </tbody>
</table>
</div></small></div>

While you will probably use inner joins the vast majority of the time (and left joins the remainder), it is helpful to understand the different types of join operations from a conceptual standpoint, and how they all fit together.

## References

- [Pandas](https://pandas.pydata.org/)
- [SQLite](https://www.sqlite.org)
- [SQLite SQL syntax](https://www.sqlite.org/lang.html)
- [Python sqlite3 library](https://docs.python.org/3/library/sqlite3.html)
