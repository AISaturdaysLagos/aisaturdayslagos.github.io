# Matrices

Vector and matrices play a central role in data science: they are probably the most common way of representing data to be analyzed and manipulated by virtually any machine learning or analytics algorithm.  However, it is also important to understand that there really two uses to matrices within data science:

1. Matrices are the "obvious" way to store tabular data (particularly when all the data is numeric) in an efficient manner.
2. Matrices are the foundation of linear algebra, which the "language" of most machine learning and analytics algorithms.

In is important to understand these two points but also the differences between them.  Under the first interpretation, matrices are essentially 2D arrays (vectors are 1D arrays, and tensors are higher-dimensional arrays); this view is fundamentally a take on how to efficiently store data in the multi-dimensional arrays.  But matrices are also the basic unit of linear algebra, which is a mathematical language for the expression and manipulation of linear systems of equations.  There are naturally overlaps between the two, but the core operations of linear algebra, such as matrix multiplication and solving linear systems of equations, are largely orthogonal to the way in which matrices are stored as arrays in memory.  Note: It is also the case that most (but definitely not all) treatments of tensors is data science actually don't do much of the second: tensors are often a nice way to store higher-dimensional tabular data, but the actual linear algebra of tensors is relatively rare despite the recent uptick of the term "tensor" in data science.

These notes will take us first through the "array" view of vectors and matrices, then to the "linear algebra" view.  We will then learn the basics of the [numpy](http://www.numpy.org/) library to manipulate numpy arrays both as arrays and as matrices.  We will finish by discussing sparse matrices, which are particularly crucial for many data science applications.

## Vectors and matrices: the "array" view

Vectors are 1D arrays of values.  We use the notation $x \in \mathbb{R}^n$ to denote that $x$ is a vector with $n$ entries and in this case the entries are all real valued.  We can consider other types of values for vectors (and in code we will commonly create vectors of Booleans or integers), but as mathematical objects it's most common to use real numbers.  We can write the elements of a vector more explicitly like so

$$
x = \left[\begin{array}{c} x_1 \\ x_2 \\ \vdots \\ x_n \end{array} \right ]
$$

where we use $x\_i$ to denote the $i$th element of the vector.  We also note that as far as the mathematical representation of vectors goes, we will consider vectors by default to be _column_ vectors (matrices with one column and many rows), as opposed to row vectors; but for example, the Numpy library doesn't make this distinction, so this will be largely for mathematical reasons.  If we want to denote a row vector, we'll use the notation $x^T$ ($x$ transposed, for the transpose operator we'll define shortly).

Matrices are 2D arrays of values, and we use the notation $A \in \mathbb{R}^{m \times n}$ to denote a matrix with $m$ rows and $n$ columns.  We can write out the elements explicitly as follows

$$
A = \left[\begin{array}{cccc}
A_{11} & A_{12} & \cdots & A_{1n} \\
A_{21} & A_{22} & \cdots & A_{2n} \\
\vdots & \vdots & \ddots & \vdots \\
A_{m1} & A_{m2} & \cdots & A_{mn}
\end{array} \right ]
$$

where $A\_ij$ denotes the entry of $A$ in the $i$th row and $j$th column.  We will use the notation $A\_{i:}$ to refer to the $i$th row of $A$ and the $A\_{:j}$ to refer to the $j$th column of $A$ (whether these represent row or column vectors depends somewhat on the situation, but we will explicitly clarify which it means whenever we use such notation).

There are also higher order generalizations of matrices (called tensors), which represented 3D or higher arrays of values.  These are in fairly common use in modern data science, though typically (but certainly not always), tensors are just used in the "multi-dimensional array" sense, not in their true linear algebra sense.  Tensors as linear operators that act on e.g. matrices or other higher-order tensors, are slightly less common most basic data science, and are a more advanced topic that is outside the scope of this course.

### Matrices for tabular data and row/column ordering

Let's start with a simple example representing tabular data using matrices, one of the more natural ways to represent such data (and as we will see in later lectures, one of the ways that lends itself well to use in machine learning).  Let's consider the "Grades" table that we previously discussed in our presentation of relational data:

| Person ID | HW1 Grade | HW2 Grade |
| :---: | :---: | :---: |
| 5 | 85 | 95 |
| 6 | 80 | 60 |
| 100 | 100 | 100 |

Ignoring the primary key column (this is not really a numeric feature, so makes less sense to store as a real-valued number), we could represent the data in the table via the matrix

$$
A \in \mathbb{R}^{3 \times 2} = \left[ \begin{array}{cc} 85 & 95 \\ 80 & 60 \\ 100 & 100 \end{array} \right ]
$$


While there seems to be little else that can be said at this point, there is actually a subtle but important point about how the data in this table is really laid out in memory.  Since data in memory is laid out sequentially (at least logically as far as programs are concerned, if not physically on the chip) we can opt to store the data in _row major order_, that is, storing each row sequentially

$$
(85, 95, 80, 60, 100, 100)
$$

or in _column major order_, storing each column sequentially

$$
(85, 80, 100, 95, 60, 100)
$$


Row major ordering is the default for 2D arrays in C (and the default for the Numpy library), while column major ordering is the default for FORTRAN.  There is no obvious reason to prefer one over the order, but due to the cache locality in CPU memory hierarchies, the different methods will be able to access the data more efficiently by row or by column respectively.  Most importantly, however, the real issue is that because a large amount of numerical code was originally (and still is) written in FORTRAN, if you ever want to call external numerical code, there is a good chance you'll need to worry about the ordering.

## Basics of linear algebra

In addition to serving as a method for storing tabular data, vector and matrices also provide a method for studying sets of linear equations.  These notes here are going to provide a very brief overview and summary of some of the primary linear algebra notation that you'll need for this course.  But it is really meant to be a refresher for those who already have some experience with linear algebra previously.  If you do not, then I have previously put out an online mini-course covering (with notes) some of the basics of linear algebra: [Linear Algebra Review](http://www.cs.cmu.edu/~zkolter/course/linalg/).  This course honestly goes a bit beyond what is needed for this particular course, but it can act a good reference.

Consider the following two linear equations in two variables $x\_1$ and $x\_2$.


$$
\begin{split}
4 x_1 - 5 x_2 & = -13 \\
-2 x_1 + 3 x_2 & = 9
\end{split}
$$


This can written compactly as the equation $A x = b$, where

$$
A \in \mathbb{R}^{2 \times 2} = \left [ \begin{array}{cc} 4 & -5 \\ -2 & 3 \end{array} \right ],
\;\; b \in \mathbb{R}^2 = \left [ \begin{array}{c} -13 \\ 9 \end{array} \right ],
\;\; x \in \mathbb{R}^2 = \left [ \begin{array}{c} x_1 \\ x_2 \end{array} \right ].
$$

As this hopefully illustrates, one of the entire points of linear algebra is to make the notation and math _simpler_ rather than more complex.  However, until you get used to the conventions, writing large equations where the size of matrices/vectors are always implicit can be a bit tricky, so you'll some care is needed to make sure you do not include any incorrect derivations.


### Basic operations and special matrices

**Addition and substraction**: Matrix addition and subtraction are applied elementwise over the matrices, and can only apply two two matrices of the same size.  That is, if $A \in \mathbb{R}^{m \times n}$ and $B \in \mathbb{R}^{m \times n}$ then their sum/difference $C = A + B$ is another matrix of the same size $C \in \mathbb{R}^{m \times n}$ where

$$
C_{ij} = A_{ij} + B_{ij}.
$$


**Transpose**: Transposing a matrix flips its rows and columns.  That is, if $A \in \mathbb{R}^n$, then it's transpose, denoted $C = A^T$ is a matrix $C \in \mathbb{R}^{n \times m}$ where

$$
C_{ij} = A_{ji}.
$$


**Matrix multiplication**: Matrix multiplication is a bit more involved.  Unlike addition and subtraction, matrix multiplication does not perform elementwise multiplication of the two matrices.  Instead, for a matrix $A \in \mathbb{R}^{m \times n}$, $B \in \mathbb{R}^{n \times p}$ (note these precise sizes, as they are important), their product $C = AB$ is a matrix $C \in \mathbb{R}^{m \times p}$ where

$$
C_{ij} = \sum_{k=1}^n A_{ik} B_{kj}.
$$

In order for this sum to make sense, notice that the number of columns in $A$ must equal the number of rows in $B$.  If you'd like a bit more of the intuition behind why matrix multiplication is defined this way, the notes above provide some important interpretations.  It's important to note the following properties, though:

* Matrix multiplication is associative: $(AB)C = A(BC)$ (i.e., it doesn't matter in what order you do the multiplications, though it _can_ matter from a computational perspective, as some orderings will be more efficient to compute than others)
* Matrix multiplication is distributive: $A(B+C) = AB + AC$
* Matrix multiplication is _not_ commutative: $AB \neq BA$. This is really true in two different ways.  Under the above matrix sizes, the multiplication $BA$ is not a valid expression if $m \neq p$ (since the number of columns in $B$ would not match the number of rows in $A$).  And even if the dimensions _do_ match (for instance if all the matrices were $n \times n$) the products will still not be equal in general.

**Identity matrix**: The identity matrix $I \in \mathbb{R}^{n \times n}$ is a square matrix with ones on the diagonal an zeros everywhere else

$$
I = \left [ \begin{array}{cccc}
1 & 0 & \cdots & 0 \\
0 & 1 & \cdots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
0 & 0 & \cdots & 1
\end{array} \right ].
$$


It has the property that for any matrix $A \in \mathbb{R}^{m \times n}$

$$
A I = I A = A
$$

where we note that the two $I$ matrices in the above equations are actually two _different_ sizes (the first one is $ n \times n$ and the second is $m \times m$, to make the math work right).  For this reason, some people use the notation $I\_n$ to explicitly denote the size of $I$, but it is not really needed, because the size that any identity must be can be inferred from the other matrices in the equation.

**Matix inverse**: For a square matrix $A \in \mathbb{R}^{n \times n}$, the matrix inverse $A^{-1} \in \mathbb{R}^{n \times n}$ is the unique matrix such that

$$
A^{-1}A = A A^{-1} = I.
$$

The matrix inverse need not exist for all square matrices (it will depend on the linear independence between rows/columns of $A$, and we will consider such possibilities a bit later in the course.

**Solving linear equations**: The matrix inverse provides an immediate method to obtain the solution to systems of linear equations.  Recall out example above of a set of linear equations $A x = b$.  If we want to find the $x$ that satisfies this equation, we multiply both sizes of the equation by $A^{-1}$ on the left to get

$$
A^{-1}A x = A^{-1}b \Longrightarrow x = A^{-1} b.
$$

The nice thing here is that as far as we are concerned in this course, the set of equations is now _solved_.  We don't have to worry about any actual operations that you may have learned about to actually obtain this solution (scaling the linear equations by some constant, adding/subtracting them to construct a solution, etc).  The linear algebra libraries we will use do all this for us, and our only concern is getting the solution into a form like the one above.

**Transpose of matrix product**: It follows immediately from the definition of matrix multiplication and the transpose that

$$
(AB)^T = B^T A^T
$$

i.e., the transpose of a matrix product is the product of the transposes, in reverse order.

**Inverse of matrix**: It also follows immediately from the definitions that for $A,B\in \mathbb{R}^{n \times n}$ both square

$$
(AB)^{-1} = B^{-1} A^{-1}
$$

i.e. the inverse of a matrix product is the product of the inverses, in reverse order.

**Inner products**: One type of matrix multiplication is common enough that it deserves special mention.  If $x,y \in \mathbb{R}^n$ are vectors of the same dimensions, then

$$
x^T y = \sum_{i=1}^n x_i y_i
$$

(the matrix product of $x$ transposed, i.e., a row vector and $y$, a column vector) is a _scalar_ quantity called the inner product of $x$ and $y$; it is simply equal to the sum of the corresponding elements of $x$ and $y$ multiplied together.

**Vector norms**: These are only slightly related to vectors matrices, but this seems like a good place to introduce it.  We will use the notation

$$
\|x\|_2 = \sqrt{x^T x} = \sqrt{\sum_{i=1}^n x_i^2}
$$

to denote the Euclidean (also called $\ell\_2$) norm of $x$.  We may occasionally also refer to the $\ell\_1$ norm

$$
\|x\|_1 = \sum_{i=1}^n |x_i|
$$

or the $\ell\_\infty$ norm

$$
\|x\|_\infty = \max_{i=1,\ldots,n} |x_i|.
$$


** Complexity of operations**:  For making efficient use of matrix operations, it is extremely important to know the big-O complexity of the different matrix operations.  Immediately from the definitions of the operations, assuming $A,B \in \mathbb{R}^{n \times n}$ and $x,y \in \mathbb{R}^n$ we have the the following complexities:

* Inner product $x^Ty$: $O(n)$
* Matrix-vector product $Ax$: $O(n^2)$
* Matrix-matrix product $AB$: $O(n^3)$
* Matrix inverse $A^{-1}$, or matrix solve $A^{-1}y$ (as we will emphasize below, these are arctually done differently; they both have the same big-O omplexity, but different concstant terms on the runtime in practice): $O(n^3)$

Note that the big-O complexity along with the associative property of matrix multiplications implies very different complexities for computing the exact same term in different ways.  For example, suppose we want to compute the matrix products $ABx$.  We could compute this as $(AB)x$ (computing the $AB$ product first, then multiplying with $x$); this approach would have complexity $O(n^3)$, as the matrix-matrix product would dominate the computation.  On the other hand, if we compute the product as $A(Bx)$ (first computing the vector product $Bx$, which produces a vector, then multiplying this by $A$), the complex is only $O(n^2)$, as we just have two matrix-vector products.  As you can imagine, these orders of operations therefore make a huge difference in terms of the time complexity of linear algebra operations.

## Numpy and software for matrices

The Numpy library is the defacto standard for manipulating matrices and vectors (and higher order tensors) from within Python.  Numpy `ndarray` objects are fundamentally multi-dimensional arrays, but the library also includes a variety of functions for processing these like matrices/vectors.  A key aspect of this library is that numpy matrices vectors are stored efficiently in memory, not via Python lists, and the operations in numpy are back by efficiently implemented linear algebra libraries.

A more complete tutorial on numpy is available here: [Numpy tutorial](https://docs.scipy.org/doc/numpy-dev/user/quickstart.html).  But these notes will introduce you to some of the basic operations you're likely to see repeatedly in many data science applications.

### Specialized linear algebra libraies

As you will hopefully appreciate throughout this course, linear algebra computations underly virtually all data science and machine learning algorithms.  Thus, making these algorithms fast is extremely important in practical applications.  And despite the seeming "simplicity" of basic linear algebra operators, the naive implementation of most algorithms will perform quite poorly.  Consider the following (in C) implementation of a matrix multiplication operator.

```c
void matmul(double **A, double **B, double **C, int m, int n, int p) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < p; j++) {
            C[i][j] = 0.0;
            for (int k = 0; k < n; k++)
                C[i][j] += A[i][k] * B[k][j];
        }
    }
}
```

In some sense, it seems like this _has_ to be the implementation of matrix multiply; it is just a simple translation of the mathematical definition to code.  But if you compiled this code, and profile it against an optimized linear algebra library, you can expect about 10 _times_ slower performance.  How is this happening?

It turns out that (precisely because linear algebra is so crucial), specialized linear algebra libraries (these include, for instance, [OpenBLAS](http://www.openblas.net/), [Intel MKL](https://software.intel.com/en-us/mkl), [ATLAS](http://math-atlas.sourceforge.net/) or [Eigen](http://eigen.tuxfamily.org/)) have gone to great lengths to exploit the custom vector processors, plus the cache hierarchy of specific architectures, to make libraries that are well-tuned to each different CPU (for example, they typically use "chunking" methods to break down matrix multiplications into smaller pieces that are better suited to exploit cache locality).  And the difference in speed between these and the more naive algorithms are extremely striking.

This is also one of the reasons why we recommend that everyone still Anaconda as their Python distribution.  Anaconda comes with versions of Numpy that are compiled with a fast linear algebra library backing it.  Chances are, if you compile your Numpy library locally, you will not link to these fast libraries, and the speed of your matrix-based code will suffer substantially from it.

To check to see if Numpy is using specialized libraries, use the command:

```python
import numpy as np
np.__config__.show()
```

<pre>
blas_mkl_info:
    libraries = [&#x27;mkl_rt&#x27;, &#x27;pthread&#x27;]
    library_dirs = [&#x27;/Users/zkolter/anaconda3/lib&#x27;]
    define_macros = [(&#x27;SCIPY_MKL_H&#x27;, None), (&#x27;HAVE_CBLAS&#x27;, None)]
    include_dirs = [&#x27;/Users/zkolter/anaconda3/include&#x27;]
blas_opt_info:
    libraries = [&#x27;mkl_rt&#x27;, &#x27;pthread&#x27;]
    library_dirs = [&#x27;/Users/zkolter/anaconda3/lib&#x27;]
    define_macros = [(&#x27;SCIPY_MKL_H&#x27;, None), (&#x27;HAVE_CBLAS&#x27;, None)]
    include_dirs = [&#x27;/Users/zkolter/anaconda3/include&#x27;]
lapack_mkl_info:
    libraries = [&#x27;mkl_rt&#x27;, &#x27;pthread&#x27;]
    library_dirs = [&#x27;/Users/zkolter/anaconda3/lib&#x27;]
    define_macros = [(&#x27;SCIPY_MKL_H&#x27;, None), (&#x27;HAVE_CBLAS&#x27;, None)]
    include_dirs = [&#x27;/Users/zkolter/anaconda3/include&#x27;]
lapack_opt_info:
    libraries = [&#x27;mkl_rt&#x27;, &#x27;pthread&#x27;]
    library_dirs = [&#x27;/Users/zkolter/anaconda3/lib&#x27;]
    define_macros = [(&#x27;SCIPY_MKL_H&#x27;, None), (&#x27;HAVE_CBLAS&#x27;, None)]
    include_dirs = [&#x27;/Users/zkolter/anaconda3/include&#x27;]

</pre>


Your output may not be exactly the same as what is shown here, but you should be able to infer from this if you're using an optimized library like (int this case), Intel MKL.

### Creating numpy arrays

The `ndarray` is the basic data type in Numpy.  These can be created the `numpy.array` command, passing a 1D list of number to create a vector or a 2D list of numbers to create an array.

```python
b = np.array([-13,9])            # 1D array construction
A = np.array([[4,-5], [-2,3]])   # 2D array contruction
print(b, "\n")
print(A)
```

<pre>
[-13   9] 

[[ 4 -5]
 [-2  3]]

</pre>


There are also special functions for creating arrays/matrices of all zeros, all ones, or of random numbers (in this case, the `np.randon.randn` create a matrix with standard random normal entries, while `np.random.rand` creates uniform random entries).

```python
print(np.ones(4), "\n")           # 1D array of ones
print(np.zeros(4), "\n")          # 1D array of zeros
print(np.random.randn(4))         # 1D array of random normal numbers
```

<pre>
[ 1.  1.  1.  1.] 

[ 0.  0.  0.  0.] 

[-0.65826018 -0.48547552 -0.12390373  0.51937501]

</pre>


```python
print(np.ones((3,4)), "\n")       # 2D array of ones
print(np.zeros((3,4)), "\n")      # 2D array of zeros
print(np.random.randn(3,4))       # 2D array of random normal numbers
```

<pre>
[[ 1.  1.  1.  1.]
 [ 1.  1.  1.  1.]
 [ 1.  1.  1.  1.]] 

[[ 0.  0.  0.  0.]
 [ 0.  0.  0.  0.]
 [ 0.  0.  0.  0.]] 

[[-0.92108207 -0.0840208  -1.49748471  0.1484692 ]
 [-0.80504092  0.47344881  0.96519561  1.02125684]
 [ 0.07350312 -0.52083043 -0.42326075  0.71938146]]

</pre>


Note that (in a design that will forever frustrate me), the size of the array is passed as a tuple to `np.ones()` and `np.zeros()`, but as a list of arguments to `np.random.randn()`.

You can also create the indentity matrix using the `np.eye()` command, and a diagonal matrix with the `np.diag()` command.

```python
print(np.eye(3),"\n")                     # create array for 3x3 identity matrix
print(np.diag(np.random.randn(3)),"\n")   # create diagonal array
```

<pre>
[[ 1.  0.  0.]
 [ 0.  1.  0.]
 [ 0.  0.  1.]] 

[[ 0.75084381  0.          0.        ]
 [ 0.         -0.17133185  0.        ]
 [ 0.          0.         -1.09201859]] 


</pre>


### Indexing into numpy arrays

You can index into Numpy arrays in many different ways.  The most common is to index into them as you would a list: using single indices and using slices, with the additional consideration that using the `:` character will select the entire span along that dimension.

```python
A = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])
print(A, "\n")
print(A[1,1],"\n")           # select singe entry
print(A[1,:],"\n")           # select entire row
print(A[1:3, :], "\n")       # slice indexing
```

<pre>
[[ 1  2  3]
 [ 4  5  6]
 [ 7  8  9]
 [10 11 12]] 

5 

[4 5 6] 

[[4 5 6]
 [7 8 9]] 


</pre>


Note the convention here in terms of the sizes returned: if we select a single entry, then we get back the value of that entry (not a 1D/2D array with just a singleton element).  If we select a single row or a single column from a 2D array we get a _1D array_ with that row or column.  And if we select a slice and/or the entire row/column along both dimensions, we get a 2D array.  This takes a while to get used to, but if, for example, we wanted to get a _2D array_ containing just the (1,1) element, we could use the code.

```python
print(A[1:2,1:2])  # Select A[1,1] as a singleton 2D array
```

<pre>
[[5]]

</pre>


Numpy also support fancier indexing with _integer_ and _Boolean_ indexing.  If we create another array or list of indices (that  is, for the rows in above array, this would be integers between 0-3 (inclusive)), then we can use this list of integers to select the rows/columns we want to include.

```python
print(A[[1,2,3],:])  # select rows 1, 2, and 3
```

<pre>
[[ 4  5  6]
 [ 7  8  9]
 [10 11 12]]

</pre>


Note that these integer indices do not need to be in order, nor do they have to include at most once instance of each row/column; we can use this notation to repeat rows/columns too.

```python
print(A[[2,1,2],:])  # select rows 2, 1, and 2 again
```

<pre>
[[7 8 9]
 [4 5 6]
 [7 8 9]]

</pre>


Note that we can also use an array of integers instead of a list, for the same purpose.

```python
print(A[np.array([2,1,2]),:])  # select rows 2, 1, and 2 again
```

<pre>
[[7 8 9]
 [4 5 6]
 [7 8 9]]

</pre>


Last, we can also use Boolean indexing.  If we specify a list or array of booleans that is the _same size_ as the corresponding row/column, then the Boolean values specify a "mask" over which values are taken.

```python
print(A[[False, True, False, True],:])  # Select 1st and 3rd rows
```

<pre>
[[ 4  5  6]
 [10 11 12]]

</pre>


As a final note, be careful if you try to use integer or boolean indexing for both dimensions.  This will attempt to select generate a 1D array of entries with both those locations.

```python
print(A[[2,1,2],[1,2,0]])    # the same as np.array([A[2,1], A[1,2], A[2,0]])
```

If you actually want to first select based upon rows, then upon columns, you'll do it like the following, essentially doing each indexing separately.

```python
A[[2,1,2],:][:,[1,2,0]]
```

### Basic operations on arrays

Arrays can be added/subtracted, multiplied/divided, and transposed, but these are _not_ all the same as their linear algebra counterparts.

```python
B = np.array([[1, 1, 1], [1,2,1], [3, 1, 3], [1, 4, 1]])
print(A+B, "\n") # add A and B elementwise (same as "standard" matrix addition)
print(A-B) # subtract B from A elementwise (same as "standard" matrix subtraction)
```

<pre>
[[ 2  3  4]
 [ 5  7  7]
 [10  9 12]
 [11 15 13]] 

[[ 0  1  2]
 [ 3  3  5]
 [ 4  7  6]
 [ 9  7 11]]

</pre>


Array multiplication and division are done _elementwise_, they are _not_ matrix multiplication or anything related to matrix inversion.

```python
print(A*B, "\n") # elementwise multiplication, _not_ matrix multiplication
print(A/B, "\n") # elementwise division, _not_ matrix inversion
```

<pre>
[[ 1  2  3]
 [ 4 10  6]
 [21  8 27]
 [10 44 12]] 

[[  1.           2.           3.        ]
 [  4.           2.5          6.        ]
 [  2.33333333   8.           3.        ]
 [ 10.           2.75        12.        ]] 


</pre>


You can transpose arrays, but note this _only_ has meaning for 2D (or higher) arrays.  Transposing a 1D array doesn't do anything, since Numpy has no notion of column vectors vs. row vectors for 1D arrays.

```python
x = np.array([1,2,3,4])
print(A.T, "\n")
print(x, "\n")
print(x.T)
```

<pre>
[[ 1  4  7 10]
 [ 2  5  8 11]
 [ 3  6  9 12]] 

[1 2 3 4] 

[1 2 3 4]

</pre>


### Broadcasting

Things start to get very fun when you add/substract/multiply/divide array of _different_ sizes.  Rather than throw an error, Numpy will try to make sense of your operation using the [Numpy broadcasting rules](https://docs.scipy.org/doc/numpy-1.13.0/user/basics.broadcasting.html).  This is an advanced topic, which often really throws off newcomers to Numpy, but with a bit of practice the rules become quite intuitive.

Broadcasting generally refers to how entries from the small array are "broadcast" to the larger array.  The basic rule is as follows: first, suppose that two Numpy arrays `A` and `B` have the same number of dimensions (e.g., they are both 2D arrays).  But suppose that along one of these dimensions `B` is only size 1. In that case, the the dimension of size 1 is `B` is automatically expanded (repeating all entries along that dimension), and then the operation between the two arrays is applied.  Let's look at a simple example:

```python
A = np.ones((4,3))          # A is 4x3
x = np.array([[1,2,3]])      # x is 1x3
A*x                          # repeat x along dimension 4 (repeat four times), and add to A
```

Effectively, the (1,3) array `x` (observer that it is actually a 2D array) is "resized" to (4,3), repeating its entries along the first dimension, and it is them multiplied elementwise by `A`.  The effective result of this is that the columns of `A` are rescaled by the values of `x`.  Note that no actual additional memory allocation happens, and the resize here is entirely from a conceptual perspective.  Alternatively, the following code would rescale the _rows_ of `A` by `x` (where here we need to construct a (4,1) sized array is order for the broadcasting to work.

```python
x = np.array([[1],[2],[3],[4]])
A*x
```

Here `x` has size (4,1), so it is effectively resized to (4,3) along the second dimension, repeating values along the columns.  This has the effect of scaling the rows of `A` by `x`.

As a final note, the rule for numpy is that if the two arrays being operated upon have _different_ numbers of dimensions, we extend the dimensions in the _leading_ dimensions to all implicitly be 1.  Thus, the following code will implicitly consider `x` (which is a 1D array of size 3), to be a (1,3) array, and then apply the broadcasting rules, which thus has the same effect as our first broadcasting example.

```python
x = np.array([1,2,3])
A*x
```

If we want to implicitly "cast" a n sized 1D array to a (n,1) sized array, we can use the notation `x[:,None]` (we put "None" for the dimensions we want to define to be 1).

```python
x = np.array([1,2,3,4])
print(x[:,None], "\n")
print(A*x[:,None])
```

<pre>
[[1]
 [2]
 [3]
 [4]] 

[[ 1.  1.  1.]
 [ 2.  2.  2.]
 [ 3.  3.  3.]
 [ 4.  4.  4.]]

</pre>


These rules can be confusing, and it takes some time to get used to them, but the advantage of broadcasting is that you can compute many operations quite efficiently, and once you get used to the notation, it is actually not the difficult to understand what is happening.  For example, from a "linear algebra" perspective, the right way to scale the column of a matrix is a matrix multiplication by a diagonal matrix, like in the following code.

```python
D = np.diag(np.array([1,2,3]))
A @ D
```

(we will cover the matrix multiplication operator `@` in a moment).  However, actually constructing the `np.diag()` matrix is wasteful: it explicitly constructs an $n \times n$ matrix that only has non-zero elements on the diagonal, then performs a dense matrix multiplication.  It is much more efficient to simply scale `A` using the broadcasting method above, as no additional storage will be allocated, and the actual scaling operation only requires $O(n^2)$ time as opposed to the $O(n^3)$ time for a full matrix multiplication.

### Linear algebra operations

Starting with Python 3, there is now a matrix multiplication operator `@` defined between numpy arrays (previously one had to use the more cumbersome `np.dot()` function to accomplish the same thing).  Note that in the following example, all the array sizes are created such that the matrix multiplications work.

```python
A = np.random.randn(5,4)
C = np.random.randn(4,3)
x = np.random.randn(4)
y = np.random.randn(5)
z = np.random.randn(4)

print(A @ C, "\n")       # matrix-matrix multiply (returns 2D array)
print(A @ x, "\n")       # matrix-vector multiply (returns 1D array)
print(x @ z)       # inner product (scalar)
```

<pre>
[[-2.349365    0.31307737  0.43701076]
 [-3.01521936  2.33512524 -0.59099322]
 [-0.02346425  0.0118288  -2.68179453]
 [ 0.58286024 -1.36334426  0.35011801]
 [ 0.56680928 -1.83411679  1.29601818]] 

[-0.19595591  0.22193364  0.88633042  0.40914083 -0.87358333] 

1.11337305084

</pre>


There is an important point to note, though, here. Depending on the sizes of the arrays passed to the `@` operator, numpy will return results of different sizes: two 2D arrays result in a 2D array (matrix-matrix product), a 2D array and a 1D array result in a 1D array (matrix-vector product), and two 1D arrays result in a scalar (just a floating point number, not an `ndarray` at all).  This can cause some issues if, for instance, your code always assumes that the result of a `@` operation between two `ndarray` objects will also return an `ndarray`: depending on the size of the arrays you pass (i.e., if they are both 1D arrays), you will actually get a floating point object, not an `ndarray` at all.

The rules can be especially confusing when we think about multiplying vectors on the left of matrices, i.e., forming a matrix-vector product $y^T A$ for $y \in \mathbb{R}^m$, $A \in \mathbb{R}^{m \times n}$.  This is a valid matrix product, but since Numpy has no distinction between column and row vectors, both the following operations compute the same 1D result (i.e., which performs the above left-multplication, but return the result just as a 1D array):

```python
print(A.T @ y, "\n")
print(y.T @ A)
```

<pre>
[-0.20969054  1.62940281 -0.89696956 -2.29205352] 

[-0.20969054  1.62940281 -0.89696956 -2.29205352]

</pre>


The confusing part is that because transposes have no meaning to for 1D arrays, the following code _also_ returns the same result, despite $y A$ not being a valid linear algebra expression.

```python
print(y @ A)
```

<pre>
[-0.20969054  1.62940281 -0.89696956 -2.29205352]

</pre>


On the other hand, trying to do the multiplication in the other order $Ay$ (which is also not a valid linear algebra expression), does throw an error.

```python
print(A @ y)
```

These are oddities that you will get used to, and while I initially thought that the notation for everything here was rather counter-intuitive, it actually does make sense (in some respect) why everything was implemented this way.

**Note**: in an attempt to "fix" these problems, the Numpy library also contains an `np.matrix` class (where everything is represented as a 2D matrix, with row and column vectors explicit, and the multiplication operator `*` overloaded to perform matrix multiplication).  Don't use this class.  The issue is that 1) when you want to perform _non-matrix_ operations, you need to cast them back to `np.ndarray` objects, which creates very cumbersome code; and 2) most function and external libraries return `ndarray` objects anyway.  There are, somewhat annoyingly, some collection of Numpy functions (and especially Scipy functions) that _do_ return `np.matrix` objects, and the one function you'll need to know is `np.asarray()`, which casts them back to arrays while not performing any matrix copies.  For example, the sparse matrix routines that we will see shortly have a function `.todense()` that returns dense version of the matrix but as an `np.matrix` object.

```python
import scipy.sparse as sp
A = sp.coo_matrix(np.eye(5))
A.todense()
```

You can cast it to an `ndarray` using the code.

```python
np.asarray(A.todense())
```

### Order of matrix multiplication operators

Let's also look at what we mentioned above, considering the order of matrix multiplications in terms of time complexity.  By default the `@` operator will be applied left-to-right, which may result is very inefficient orders for the matrix multiplication.

```python
A = np.random.randn(1000,1000)
B = np.random.randn(1000,2000)
x = np.random.randn(2000)
%timeit A @ B @ x
```

<pre>
67.9 ms ± 3.91 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)

</pre>


This performs the matrix products $(AB)x$, which computes the inefficient matrix multiplication first.  If we want to compute the product in the much more efficient order $A(Bx)$, we would use the command

```python
%timeit A @ (B @ x)
```

<pre>
1.44 ms ± 73.3 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)

</pre>


The later operation can be about 50x faster that the first version, and the difference only gets larger for larger matrices.  Be _very_ careful about this point when you are multiplying big matrices and vectors together.

### Inverses and linear solves

Finally, Numpy includes the routine `np.linalg.inv()` for computing the matrix inverse $A^{-1}$ and `np.linalg.solve()` for computing the matrix solve $A^{-1}b$, for $A \in \mathbb{R}^{n \times n}$, $b \in \mathbb{R}^n$.

```python
b = np.array([-13,9])
A = np.array([[4,-5], [-2,3]])

print(np.linalg.inv(A), "\n")   # explicitly form inverse
print(np.linalg.solve(A,b))     # compute solution A^{-1}b
```

<pre>
[[ 1.5  2.5]
 [ 1.   2. ]] 

[ 3.  5.]

</pre>


Obviously the `np.linalg.solve()` routine is also equivalent to the matrix-vector product with the inverse.

```python
print(np.linalg.inv(A) @ b)    # don't do this
```

<pre>
[ 3.  5.]

</pre>


However, you should _not_ do this.  In general, actually computing the inverse and then multiplying by a vector is both slower and less numerically stable than just solving the linear system.  For those who are curious (you won't need to know this for this class, but it can be useful to understand), this is because performing the solve internally actually computes a _factorization_ of the matrix called the LU factorization: it decomposes $A$ into the matrix product $A = LU$ where $L$ is a lower triangular matrix (all entries above the diagonal are zero), and $U$ is an upper triangular matrix (all entries below the diagonal are zero); if we want to be even more precise it actually computes the LU factorization on a version of $A$ with permuted rows and columns, but that is definitely beyond the scope of this course.  After factorizing $A$ in this manner, it computes the inverse

$$
A^{-1} b = (L U)^{-1} b = U^{-1} L^{-1} b
$$

In turns out that computing the product $L^{-1} b$ (again, not actually explicitly computing the inverse of $A$, just computing the solve) for a triangular matrix is very efficient, it just takes $O(n^2)$ operations instead of $O(n^3)$ operations as for a generic matrix.  This means that once we compute the factorization $A = LU$ (which does itself take $O(n^3)$ time, as a note, which is why matrix solves are $O(n^3)$ complexity), solving for the right hand size $b$ is "easy".

In fact the way that we compute the inverse is by first computing the LU factorization and then "solving" for the right hand side $I$.  But obviously if we just want to ultimately solve for a single right hand size $b$, this is an unnecessary step, and it introduces additional error into the computation.  For this reason, you will almost ways prefer to use `np.linalg.solve()` unless to really need elements of the inverse itself (and not just to multiply the inverse by some expression).


## Sparse matrices
The last topic we will touch on, which at our level of discussion will relate more to the representation of matrices than the linear algebra behind it, is the concept of sparse matrices.  Many problems in data science deal with matrix representations that are inherently _sparse_ (that is, they contain a majority of zero elements, with only a few non-zeros).  For example, in the two forthcoming lectures, we will discuss graphs and free text modeling.  As we will see, a primary method for representing graphs will be the adjacency matrix, a matrix that has as a non-zero element in location $i,j$ if there is an edge between node $i$ and node $j$; many natural graphs have only a small number of nodes connected to each other, so the resulting adjacency matrix is sparse.  Similarly, in free text modeling it is common to represent documents via a "bag of words" model, where we represent each document as a large vector that that indicates which words occur in the document; since typical documents only contain a small fraction of all possible words, this "word-document co-occurrence" matrix is usually sparse.

While of course it is possible to represent a sparse matrix using the standard dense format specified above, doing so is quite wasteful from both a space and time complexity standpoint; explicitly representing (and multiplying by) many zeros is both a waste of memory and computation.  Because of this, when matrices have a significant number of zero entries, it makes sense that we should somehow exploit this fact to reduce memory consumption and speed up computation.  Fortunately, there are a large collection of sparse matrix libraries that accomplish these goals exactly.  While we of course won't have the ability to go into how these libraries work in any real detail (especially for the more complex operations like sparse matrix-matrix multiplies (already hard enough of a problem to do "easily") let alone more complex operations like sparse matrix factorizations (where there is a whole subfield dedicated to such algorithms).  The good news, though, is that from a pure mathematical standpoint, sparse matrices naturally behave just like dense matrices (matrix multiplication, inversion, etc, all have the same syntax); really, these are fundamentally a _computational_ tool, and with a slight understanding of how sparse matrices function, you can make fairly good use of the libraries even if you don't initially know all the details behind the algorithms.

### Sparse data formats
The first and primary element we will highlight is the _data formats_ used to specify sparse matrices.  Unlike dense matrices, where there is a natural way to represent elements in memory (though even there we had the column-major/row-major choice), there are many different ways we can represent sparse matrices, and different approaches are better suited to different tasks.  Understanding these formats is a starting point to learning to use sparse matrix libraries, and will also draw connections to later topics we discuss.  In these notes we will cover two of the most common formats you'll encounter: the coordinate (COO) format, and the compressed sparse column (CSC) format.  Most other formats have some similarity to these, so they serve as a good starting point to understanding sparse matrices.

**Coordinate (COO) format.** Perhaps the most natural way to express a sparse matrix is by a "list" of (row, column, value) tuples; because we know that only very few locations will actually have non-zero entries, this format lets us just express the entries that exist, where all other entries that are not specified are assumed to be zero.  This is in fact essentially exactly the intuition behind the coordinate (commonly abbreviated COO, despite the fact that it is not an acronym) format for sparse matrices.  In coo format, we use three 1D arrays to specify a matrix, a `row_indices` array, a `col_indices` array, and a `values` array.  These contain exactly what you would expect.  For each non-zero entry in the matrix is there a corresponding element in these vectors that specifies the row, column, and value of that matrix.  Let's consider the example matrix

$$
A \in \mathbb{R}^{4\times 4} = \left [ \begin{array}{cccc}
0 & 0 & 3 & 0 \\
2 & 0 & 0 & 1 \\
0 & 1 & 0 & 0 \\
4 & 0 & 1 & 0
\end{array} \right ].
$$

A COO representation of this matrix could be (this is just specifying generic arrays, not in any particular language):

```python
values = [2, 4, 1, 3, 1, 1]
row_indices = [1, 3, 2, 0, 3, 1]
column_indices = [0, 0, 1, 2, 2, 3]
```

Although here we ordered the entries in these arrays in a "column-major" fashion, COO format has no such requirements: we could specify the elements of the matrix in any order we desire (and even have duplicate entires, where the true value is assumed to be the sum of all entries correspond to a particular row and column).  The size of all these entries will be equal to the number of nonzeros in `A`, a number which will refer to as $\mathrm{nnz}$ when we need to use it notationally.  Finally, not that in addition to the entries in the above matrices, we would also need to store the actual size of the array (number of rows $m$ and columns $n$ respectively), to cover the  case where we have additional rows/columns of all zeros.

The advantage of the COO format is that it is good for constructing sparse matrices; because items can be stored in any order to add a new element the matrix, we simply append an entry to the arrays (depending on how these arrays are stored, you would typically "pre-allocate" a bit of extra storage, resizing the arrays as needed, just like Python does with its lists, if you expect to be frequently adding elements).  On the flipside, it is quite back for _inspecting_ sparse matrices.  Suppose we wanted to look up the value `A[i,j]` in a sparse matrix in COO format.  Because elements can be in any order, we would have no option but to perform a linear scan through the entire arrays, to see if any row/column pair matched `i`/`j`, with complexity $O(n)$ (where here say that the matrix is $n \times n$).

If the matrix in COO format was always maintained in column-major order, as described above, then this complexity could be reduced to $O(\log n)$ via binary search on the column indices; however, maintaining it in this order would also remove our ability to easily add new elements to matrix, because we would need to shift items in the various arrays (i.e., by copying the memory to a later location) whenever a new element was added.  And if we're willing to make this sacrifice, it turns out there is a better way to store the matrix, described next.

**Compressed sparse column (CSC) format.** The downsides of the above approach motivate a different matrix storage, where we _do_ explicitly maintain a column-major ordering of the non-zero entries.  However, if we are going to do so, then it turns out we actually get a more efficient structure if we change the nature of the `column_indices` array.  Instead of an $\mathrm{nnz}$-dimensional array containing the column index of each entry, we make the array be a $(n+1)$-dimenional array (remember that $n$ is the number of columns in the matrix), pointing to the _index_ of the starting location for each column in the `row_indices` and `values` array.

This is a bit hard to understand at first, so an example can make things more understandable.  Instead of the COO format above, the CSC format consists of the arrays

```python
values = [2, 4, 1, 3, 1, 1]
row_indices = [1, 3, 2, 0, 3, 1]
column_indices = [0, 2, 3, 5, 6]
```

Let's unpack this a bit.  The fact that `column_indices` has a 5 in element 3 (remember, we are assuming zero indexing), means that the index-3 column (really the forth column) starts at index 5 in the `row_indices` and `values` arrays; the fact that `column_indices` has a 2 in element 1 means that the index-1 column (really the second column) starts at index 2 in the `row_indices` and `values` arrays.

The advantage to this format is that it is extremely efficient to look up _all_ the entries in a given column $i$.  If we want to know the entries of column `i`, we would use simply get the slice (using Python notation here)

```python
values[column_indices[i]:column_indices[i+1]]
```

(the same could be done to get the row indices).  This also hopefully clarifies why the `column_indices` array contains $n+1$ entries: so that we can always use the range `column_indices[i]:column_indices[i+1]` to get all the items in the column (the last entry in `column_indices` must therefore be equal to the number of non-zero elements in the matrix.

As is hopefully apparent, CSC format is typically much better than COO for quickly accessing elements, especially accessing single rows in the matrix (there is a corresponding compressed sparse row (CSR) that does the exact same thing but row-wise, for quick access to rows).  But conversely, it is very poor at adding new elements to the array (this requires shifting all subsequent items in the `row_indices` and `values` columns, and incrementing subsequent elements in the `column_indices` array).

### Basics of sparse matrix computation

The above discussion hopefully makes it fairly obvious why sparse matrices are a good idea from a storage standpoint: instead of storing $mn$ elements for an $m \times n$, we can store $3\cdot\mathrm{nnz}$ arrays (we ignore storing $m$ and $n$, because you actually need to do this for both sparse and dense matrices anyway) for COO format, or $2\cdot\mathrm{nnz} + n + 1$ elements for CSC format.

But why are sparse matrices also more computationally efficient.  To give a brief sense of this, let's consider one of the most ubiquitous operations in linear algebra, the product of a (dense or sparse) matrix with a (dense) vector $Ax$ for $A \in \mathbb{R}^{m \times n}$, $x \in \mathbb{R}^n$.  This operation will be $O(mn)$ if we represent $A$ densely (this follows immediately from the definition of the matrix-vector product).  But what about if we represent $A$ COO sparse format.  In this case, the algorithm for matrix-vector products is fairly simple

1. Initialize $y = 0$
2. For each $(i,j,v)$ in $A$:
    * Set $y\_i := y\_i + x\_j \cdot v$

After running the algorithm, $y = Ax$, as this is exactly equivalent to the traditional definition of matrix-vector products, we are just accumulating the sum for each individual tuple in $A$ represented in COO form (if the above algorithm isn't apparent to you, try to derive it directly from the definition of the COO format and the definition of matrix multiplication above).  Beacuse this algorithm obviously only loops over the non-zero entries of $A$ once, it rquires only $O(\mathrm{nnz})$ operations, a potentially significant reduction.

We can also write a similar routine for matrices in CSC format, though we leave that out here for simplicit. As a general rule of thumb, you should use sparse matrices for matrix-vector multiplication if your matrix is 80% zero or (there due to the 2-3x more storage needed, and the fact that the matrix-vector products are usually a little less cache efficient than purely dense operations).  If your operations involve matrix solves, then a better rule of this is that you need 95% sparsity or more (operations related to matrix solves actually decrease sparsity of the intermediate components we compute).  This last rule of thumb, though, actually depends on the precise sparsity pattern of the data (if the sparsity is truly random, it woudl actually require much higher degrees of sparsity to make matrix sovles worth it, but for reasons we won't go into, many "realstic" matrices have structure that makes the sparse solve more efficient than the worst case.

### Python sparse matrix libraries

The standard library for manipulating sparse matrices with Python is the [scipy.sparse](https://docs.scipy.org/doc/scipy/reference/sparse.html) module.  The library supports several different types of sparse matrix formats (along with operations for converting betweeen the different types), and interfaces with well-established third-party code for all needed linear algebra functions.

Let's briefly look at how to construct a sparse matrix in `scipy.sparse` (we can convert to a dense matrix by calling the `.todense()` function, though of course this should only be done on small instances).

```python
import scipy.sparse as sp


values = [2, 4, 1, 3, 1, 1]
row_indices = [1, 3, 2, 0, 3, 1]
column_indices = [0, 0, 1, 2, 2, 3]
A = sp.coo_matrix((values, (row_indices, column_indices)), shape=(4,4))
print(A.todense())
```

<pre>
[[0 0 3 0]
 [2 0 0 1]
 [0 1 0 0]
 [4 0 1 0]]

</pre>


We can directly access the values, rows indices, and column indices of a COO sparse matrix via the `.data`, `.row`, and `.col` properties respectively (indeed, this is exactly how the sparse matrix is represented internally, with these three attributes).  Each of these are a 1D numpy array that store the data for the matrix.

```python
print(A.data)
print(A.row)
print(A.col)
```

<pre>
[2 4 1 3 1 1]
[1 3 2 0 3 1]
[0 0 1 2 2 3]

</pre>


We can also easily convert to CSC format.

```python
B = A.tocsc()
```

For a CSC matrix, the values are in still in the `.data` pointer, but the row indices and column indices (as we used the terms), are in the `.indices` and `.indptr` arrays; again, this are all just numpy arrays, that internally store the actual data of the matrix.

```python
print(B.data)
print(B.indices)
print(B.indptr)
```

<pre>
[2 4 1 3 1 1]
[1 3 2 0 3 1]
[0 2 3 5 6]

</pre>


As a final example, let's create a 1000x1000 sparse matrix with 99.9% sparsity plus an identity matrix (we'll do this with the `sp.rand` call, which randomly chooses entries to fill in, and then makes samples them from a uniform distribution ... we add the identity to make the matrix likely to be invertible).  The precise nature of the matrix isn't important here, we just want to consider the timing.

```python
A = sp.rand(1000,1000, 0.001) + sp.eye(1000)
B = np.asarray(A.todense())
x = np.random.randn(1000)
%timeit A @ x
%timeit B @ x
```

<pre>
12.8 µs ± 2.57 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each)
501 µs ± 73.9 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)

</pre>


Here the sparse version is about 50x faster, though of course the speedup will increase with sparsity relative to the dense matrix.

```python
import scipy.sparse.linalg as spla
A = A.tocsc()
%timeit spla.spsolve(A,x)     # only works with CSC or CSR format
%timeit np.linalg.solve(B,x)
```

<pre>
1.04 ms ± 9.63 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)
23.8 ms ± 2.39 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)

</pre>


Similarly, the sparse version is about 20x faster.  You can try to experiment to see where the break-even point for the sparse/dense tradeoff is, but as mentioned above, for matrix inverses this is going to be very problem specific, so you won't get too much insight until you start using real data.

## References

- [Numpy library](http://www.numpy.org)
- [Linear Algebra Review](http://www.cs.cmu.edu/~zkolter/course/linalg/)
- [Numpy broadcasting rules](https://docs.scipy.org/doc/numpy-1.13.0/user/basics.broadcasting.html)
- [Scipy sparse module](https://docs.scipy.org/doc/scipy/reference/sparse.html)
