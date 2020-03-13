# Recitation 4

### Homework Tips

* Be familiar with the [What Can I Ask On Diderot?](www.diderot.one/course/10/dosts/?is_inbox=yes&dost=4658) policy
* Talk to other students taking the course -- they can help you and you can help them.
* Feel free to meet other students during the Collaboration Space -- every Tuesday in GHC 4303.
* Look for the "Common Problems in Homework x" post on Diderot before asking questions online.

### TA Hours

* Come this week! Don't wait until the last week.

```python
import numpy as np
import scipy.sparse as sp
from IPython.display import HTML, display
import tabulate
```

## Numpy Broadcasting

Numpy has some magic that allows you to perform operations with arrays of different shapes. When working with more advanced operations, understanding broadcasting is important to making sure you don't accidentally break something. We're going to explore those rules in a simpler manner than [the official tutorial](https://numpy.org/devdocs/user/theory.broadcasting.html).

Here some pseudocode expresses these rules:

```
function broadcast(a, b):
    if a.ndim != b.ndim:
        add one-element dimensions to a or b until they have the same number of dimensions

    for i in range(a.ndim):
        if a.shape[i] == b.shape[i]:
            both dimensions are the same length, no broadcasting necessary
        elif a.shape[i] == 1:
            copy dimension [i, i+1, ...] of a until it is the same as b
        elif b.shape[i] == 1:
            copy dimension [i, i+1, ...] of b until it is the same as a
        else:
            raise "ValueError: operands could not be broadcast together"
```

(Its not actually implemented like this. See the official docs for details.)

Note that the dimensions are not actually copied; Numpy does something that is far more efficient than that.

From these simple rules, we get some pretty interesting behaviour:

```python
def pp(a):
    if a.ndim < 2:
        a = [a]
    display(HTML(tabulate.tabulate(a, tablefmt='html')))
    
def rnd(*a):
    return np.random.permutation(np.prod(a)).reshape(a)
```

```python
# Constant * Array:
b = rnd(5, 3)
pp(b)
```

<div><small><table>
<tbody>
<tr><td style="text-align: right;"> 3</td><td style="text-align: right;">10</td><td style="text-align: right;">11</td></tr>
<tr><td style="text-align: right;"> 7</td><td style="text-align: right;"> 2</td><td style="text-align: right;"> 0</td></tr>
<tr><td style="text-align: right;"> 1</td><td style="text-align: right;"> 8</td><td style="text-align: right;"> 9</td></tr>
<tr><td style="text-align: right;">13</td><td style="text-align: right;"> 4</td><td style="text-align: right;">12</td></tr>
<tr><td style="text-align: right;">14</td><td style="text-align: right;"> 5</td><td style="text-align: right;"> 6</td></tr>
</tbody>
</table></small></div>

```python
pp(2 + b)
```

<div><small><table>
<tbody>
<tr><td style="text-align: right;"> 5</td><td style="text-align: right;">12</td><td style="text-align: right;">13</td></tr>
<tr><td style="text-align: right;"> 9</td><td style="text-align: right;"> 4</td><td style="text-align: right;"> 2</td></tr>
<tr><td style="text-align: right;"> 3</td><td style="text-align: right;">10</td><td style="text-align: right;">11</td></tr>
<tr><td style="text-align: right;">15</td><td style="text-align: right;"> 6</td><td style="text-align: right;">14</td></tr>
<tr><td style="text-align: right;">16</td><td style="text-align: right;"> 7</td><td style="text-align: right;"> 8</td></tr>
</tbody>
</table></small></div>

Let's use the pseudocode to examine why this works. We begin with:

    a = []     array of [2]
    b = [5, 3] array of [0..14]

Numpy treats the constant like a zero-dimensional array. In the first step, it observes that there are fewer dimensions in `a` than `b`, so pads `a` with size-1 dimensions:

    a = [1, 1] array of [2]
    b = [5, 3] array of [0..14]

Now working from left-to-right, it examines the first dimension and notices that they are of different sizes, so broadcasting is necesssary. Because the first dimension of `a` is 1, it can be broadcast to match the first dimension of `b`. After this step, the array now looks like:

    a = [5, 1] array of [2, 2, 2, 2, 2]
    b = [5, 3] array of [0..14]

Now it examines the second dimension. Following the same pattern as before, it broadcasts the second dimension of `a` to match `b`:

    a = [5, 3] array of [2, 2,...]
    b = [5, 3] array of [0..14]

Now that the arrays are the same size, they can be added element-by-element.

Lets try this again, this time with a 1-d array to a 2-d array:

```python
c = rnd(3)
pp(c)
```

<div><small><table>
<tbody>
<tr><td style="text-align: right;">2</td><td style="text-align: right;">0</td><td style="text-align: right;">1</td></tr>
</tbody>
</table></small></div>

```python
pp(b + c)
```

<div><small><table>
<tbody>
<tr><td style="text-align: right;"> 5</td><td style="text-align: right;">10</td><td style="text-align: right;">12</td></tr>
<tr><td style="text-align: right;"> 9</td><td style="text-align: right;"> 2</td><td style="text-align: right;"> 1</td></tr>
<tr><td style="text-align: right;"> 3</td><td style="text-align: right;"> 8</td><td style="text-align: right;">10</td></tr>
<tr><td style="text-align: right;">15</td><td style="text-align: right;"> 4</td><td style="text-align: right;">13</td></tr>
<tr><td style="text-align: right;">16</td><td style="text-align: right;"> 5</td><td style="text-align: right;"> 7</td></tr>
</tbody>
</table></small></div>

Lets look at this again. We start with:

    c = [3]    array of [0..2]
    b = [5, 3] array of [0..14]

After adding padding dimensions:

    c = [1, 3] array of [0..2]
    b = [5, 3] array of [0..14]

We broadcast the first dimension of `c` to match `b`:

    c = [5, 3] array of [0..2]
    b = [5, 3] array of [0..14]

And now it can be added!

What if we want to add over the columns instead?

```python
d = rnd(5)
pp(d)
```

<div><small><table>
<tbody>
<tr><td style="text-align: right;">3</td><td style="text-align: right;">0</td><td style="text-align: right;">1</td><td style="text-align: right;">2</td><td style="text-align: right;">4</td></tr>
</tbody>
</table></small></div>

```python
pp(b + d)
```

The trivial way of doing this doesn't work. We start with:

    d = [5]    array of [0..4]
    b = [5, 3] array of [0..14]

After adding padding dimensions:

    c = [1, 5] array of [0..4]
    b = [5, 3] array of [0..14]

We broadcast the first dimension of `d` to match `b`:

    c = [5, 5] array of [0..4]
    b = [5, 3] array of [0..14]

And the second dimension does not agree, so they can't be broadcast.

To get it to broadcast correctly, we need to add the padding dimension in the right place ourselves. Here's an example:

```python
e = d[:,None]
pp(e)
print(e.shape)
```

<div><small><table>
<tbody>
<tr><td style="text-align: right;">3</td></tr>
<tr><td style="text-align: right;">0</td></tr>
<tr><td style="text-align: right;">1</td></tr>
<tr><td style="text-align: right;">2</td></tr>
<tr><td style="text-align: right;">4</td></tr>
</tbody>
</table></small></div>

The trivial way of doing this doesn't work. We start with:

    e = [5, 1] array of [0..4]
    b = [5, 3] array of [0..14]

No padding dimensions need to be added here, and the first dimensions already agree. We broadcast the _second_ dimension of `e` to match `b`:

    e = [5, 3] array of [0..4]
    b = [5, 3] array of [0..14]

And now they can be added.

```python
pp(b + e)
```

<div><small><table>
<tbody>
<tr><td style="text-align: right;"> 6</td><td style="text-align: right;">13</td><td style="text-align: right;">14</td></tr>
<tr><td style="text-align: right;"> 7</td><td style="text-align: right;"> 2</td><td style="text-align: right;"> 0</td></tr>
<tr><td style="text-align: right;"> 2</td><td style="text-align: right;"> 9</td><td style="text-align: right;">10</td></tr>
<tr><td style="text-align: right;">15</td><td style="text-align: right;"> 6</td><td style="text-align: right;">14</td></tr>
<tr><td style="text-align: right;">18</td><td style="text-align: right;"> 9</td><td style="text-align: right;">10</td></tr>
</tbody>
</table></small></div>

Broadcasting can give you some pretty awesome results. Here's an example of generating a two-dimensional array using two one-dimensional arrays. Try using our analysis to understand why this works the way it does; the official documentation explains [this example](https://numpy.org/devdocs/user/theory.broadcasting.html).

```python
p = np.array([0, 1, 2, 3, 4, 5])
q = np.array([0, 10, 20])

pp(p[:,None] + q)
```

<div><small><table>
<tbody>
<tr><td style="text-align: right;">0</td><td style="text-align: right;">10</td><td style="text-align: right;">20</td></tr>
<tr><td style="text-align: right;">1</td><td style="text-align: right;">11</td><td style="text-align: right;">21</td></tr>
<tr><td style="text-align: right;">2</td><td style="text-align: right;">12</td><td style="text-align: right;">22</td></tr>
<tr><td style="text-align: right;">3</td><td style="text-align: right;">13</td><td style="text-align: right;">23</td></tr>
<tr><td style="text-align: right;">4</td><td style="text-align: right;">14</td><td style="text-align: right;">24</td></tr>
<tr><td style="text-align: right;">5</td><td style="text-align: right;">15</td><td style="text-align: right;">25</td></tr>
</tbody>
</table></small></div>

## Multiplications in Numpy

This is a common source of difficult. We're going to go over each type of multiplication (both dense and sparse) and explain how its done.

Lets begin with the most important caveat: __`np.array`, `np.matrix`, and `sp.sparse.*` handle multiplication slighly differently!__

And here are some tips:
 1. When working with sparse matrices, you should use Scipy operations (not Numpy ones)
 2. When working with dense matrices, you can use either operation.
 3. When working with `np.array` objects:
     - `a * b` is equivalent to `np.multiply(a, b)`
     - `a @ b` is equivalent to `np.matmul(a, b)`.
 4. When working with `np.matrix` objects:
     - `a * b` is equivalent to `np.matmul(a, b)`, and
     - `a @ b` is equivalent to `np.matmul(a, b)`.
 5. The `matmul` operation has [special broadcasting rules](https://docs.scipy.org/doc/numpy/reference/generated/numpy.matmul.html).
 6. The syntax `(5,)` means a tuple of length 1 with the value `5`.
 7. For `np.multiply`, the order of arguments does not matter.

When working with these, it is important to always know what type of array or matrix you are dealing with and what the size is.

Lets begin with the simplest operation:

```python
a = np.array(5)
b = np.array(3)

np.multiply(a, b) == a.dot(b) == 15
```

Scalars can be multiplied regularly; using matrix multiplication operations on them throws an error.

### 1-D `array`:

Now we'll scale this up to 1-d arrays.

```python
c = rnd(5)
d = rnd(5)
pp(c)
pp(d)
```

<div><small><table>
<tbody>
<tr><td style="text-align: right;">0</td><td style="text-align: right;">1</td><td style="text-align: right;">2</td><td style="text-align: right;">4</td><td style="text-align: right;">3</td></tr>
</tbody>
</table></small></div>

The `*` operation (equivalent to `np.multiply`) calculates the element-wise product of the two arrays. (This is sometimes called the Hadamard product.)

```python
pp(c*d)
```

<div><small><table>
<tbody>
<tr><td style="text-align: right;">0</td><td style="text-align: right;">4</td><td style="text-align: right;">2</td><td style="text-align: right;">8</td><td style="text-align: right;">0</td></tr>
</tbody>
</table></small></div>

The dot product calculates the sum of the element-wise product:

```python
c.dot(d)
```

As does the matrix product:

```python
c @ d
```

#### Transpose

The transpose of a 1-d array does not change the array.

```python
print(c.shape)
print(c.T.shape)
```

<pre>
(5,)
(5,)

</pre>


If you want to convert it to a row-array, you can do it like this:

```python
# Add an extra dimension and then transpose:
print(c[:,None].T.shape)

# Or just make a new dimension in the right place:
print(c[None,:].shape)
```

<pre>
(1, 5)
(1, 5)

</pre>


### Row- and column-`array`

Working with the transpose of c, we see that `dot`, `multiply`/`*`, and `matmul`/`@` give the same values, but with an extra dimension:

```python
e = c[:,None] # e is c with an extra dimension.

e.T.dot(d)
```

```python
e.T * d
```

```python
np.matmul(e.T, d)
```

You can also get the cross product of two vectors using matrix multiplication:

```python
pp(c[:, None].T @ d[:, None])
```

<div><small><table>
<tbody>
<tr><td style="text-align: right;">14</td></tr>
</tbody>
</table></small></div>

```python
pp(c[:, None] @ d[:, None].T)
```

<div><small><table>
<tbody>
<tr><td style="text-align: right;">0</td><td style="text-align: right;">3</td><td style="text-align: right;">6</td><td style="text-align: right;">12</td><td style="text-align: right;"> 9</td></tr>
<tr><td style="text-align: right;">0</td><td style="text-align: right;">4</td><td style="text-align: right;">8</td><td style="text-align: right;">16</td><td style="text-align: right;">12</td></tr>
<tr><td style="text-align: right;">0</td><td style="text-align: right;">1</td><td style="text-align: right;">2</td><td style="text-align: right;"> 4</td><td style="text-align: right;"> 3</td></tr>
<tr><td style="text-align: right;">0</td><td style="text-align: right;">2</td><td style="text-align: right;">4</td><td style="text-align: right;"> 8</td><td style="text-align: right;"> 6</td></tr>
<tr><td style="text-align: right;">0</td><td style="text-align: right;">0</td><td style="text-align: right;">0</td><td style="text-align: right;"> 0</td><td style="text-align: right;"> 0</td></tr>
</tbody>
</table></small></div>

```python
print(c[:,None].shape)
c.T.T @ d
```

#### Caveat: `np.matrix`

The `*` operation is equivalent to `np.matmul`.

```python
np.matrix(c).T * d
```

### 1-d and 2-d `array`

We begin with a (3, 5) array:

```python
f = rnd(3, 5)
pp(f)
```

<div><small><table>
<tbody>
<tr><td style="text-align: right;"> 5</td><td style="text-align: right;">2</td><td style="text-align: right;">10</td><td style="text-align: right;"> 7</td><td style="text-align: right;"> 9</td></tr>
<tr><td style="text-align: right;"> 0</td><td style="text-align: right;">4</td><td style="text-align: right;"> 8</td><td style="text-align: right;">14</td><td style="text-align: right;">13</td></tr>
<tr><td style="text-align: right;">12</td><td style="text-align: right;">1</td><td style="text-align: right;"> 6</td><td style="text-align: right;">11</td><td style="text-align: right;"> 3</td></tr>
</tbody>
</table></small></div>

Recall the shape of `c` is (5)

```python
pp(c)
```

<div><small><table>
<tbody>
<tr><td style="text-align: right;">3</td><td style="text-align: right;">4</td><td style="text-align: right;">1</td><td style="text-align: right;">2</td><td style="text-align: right;">0</td></tr>
</tbody>
</table></small></div>

#### Matrix-vector product

You can also perform matrix multiplication or the `dot` operation between the (3, 5) `f` and (5,) `c` to get a result of shape (3,):

```python
pp(f @ c)
```

<div><small><table>
<tbody>
<tr><td style="text-align: right;">47</td><td style="text-align: right;">52</td><td style="text-align: right;">68</td></tr>
</tbody>
</table></small></div>

```python
pp(f.dot(c))
```

<div><small><table>
<tbody>
<tr><td style="text-align: right;">47</td><td style="text-align: right;">52</td><td style="text-align: right;">68</td></tr>
</tbody>
</table></small></div>

#### Elementwise product by rows

The broadcast rules will automatically broadcast `c` from (5) to (3, 5) by copying the rows; this results in each row of `f` being multiplied element-wise by `c`.

```python
pp(f * c)
```

<div><small><table>
<tbody>
<tr><td style="text-align: right;">15</td><td style="text-align: right;"> 8</td><td style="text-align: right;">10</td><td style="text-align: right;">14</td><td style="text-align: right;">0</td></tr>
<tr><td style="text-align: right;"> 0</td><td style="text-align: right;">16</td><td style="text-align: right;"> 8</td><td style="text-align: right;">28</td><td style="text-align: right;">0</td></tr>
<tr><td style="text-align: right;">36</td><td style="text-align: right;"> 4</td><td style="text-align: right;"> 6</td><td style="text-align: right;">22</td><td style="text-align: right;">0</td></tr>
</tbody>
</table></small></div>

#### Elementwise product by columns

If you want to multiply the elements each column of `f` by a corresponding element in an array `g`, the easiest way is to explicitly create a dimension of size 1 so the broadcasting engine knows to copy along that dimension.

```python
g = np.array([1, 2, 3])
pp(g)
```

<div><small><table>
<tbody>
<tr><td style="text-align: right;">1</td><td style="text-align: right;">2</td><td style="text-align: right;">3</td></tr>
</tbody>
</table></small></div>

```python
pp(f * g[:, None])
```

<div><small><table>
<tbody>
<tr><td style="text-align: right;"> 5</td><td style="text-align: right;">2</td><td style="text-align: right;">10</td><td style="text-align: right;"> 7</td><td style="text-align: right;"> 9</td></tr>
<tr><td style="text-align: right;"> 0</td><td style="text-align: right;">8</td><td style="text-align: right;">16</td><td style="text-align: right;">28</td><td style="text-align: right;">26</td></tr>
<tr><td style="text-align: right;">36</td><td style="text-align: right;">3</td><td style="text-align: right;">18</td><td style="text-align: right;">33</td><td style="text-align: right;"> 9</td></tr>
</tbody>
</table></small></div>

This works because we're multiplying `f` (3, 5) by `g[:, None]` (3, 1), shapes that can be broadcast together by the rules we've discussed. (This also works when the LHS is sparse and the RHS is either sparse or dense.)

Another trick to do this is to create a diagonal matrix and left-multiply with that. You have to use matrix multiplication for that operation.

```python
pp(np.diag(g))
```

<div><small><table>
<tbody>
<tr><td style="text-align: right;">1</td><td style="text-align: right;">0</td><td style="text-align: right;">0</td></tr>
<tr><td style="text-align: right;">0</td><td style="text-align: right;">2</td><td style="text-align: right;">0</td></tr>
<tr><td style="text-align: right;">0</td><td style="text-align: right;">0</td><td style="text-align: right;">3</td></tr>
</tbody>
</table></small></div>

```python
pp(np.diag(g) @ f)
```

<div><small><table>
<tbody>
<tr><td style="text-align: right;"> 5</td><td style="text-align: right;">2</td><td style="text-align: right;">10</td><td style="text-align: right;"> 7</td><td style="text-align: right;"> 9</td></tr>
<tr><td style="text-align: right;"> 0</td><td style="text-align: right;">8</td><td style="text-align: right;">16</td><td style="text-align: right;">28</td><td style="text-align: right;">26</td></tr>
<tr><td style="text-align: right;">36</td><td style="text-align: right;">3</td><td style="text-align: right;">18</td><td style="text-align: right;">33</td><td style="text-align: right;"> 9</td></tr>
</tbody>
</table></small></div>

This can be made reasonably efficient using a sparse diagonal matrix, which only includes the entries along the diagonal:

```python
pp(sp.diags(g) @ f)
```

<div><small><table>
<tbody>
<tr><td style="text-align: right;"> 5</td><td style="text-align: right;">2</td><td style="text-align: right;">10</td><td style="text-align: right;"> 7</td><td style="text-align: right;"> 9</td></tr>
<tr><td style="text-align: right;"> 0</td><td style="text-align: right;">8</td><td style="text-align: right;">16</td><td style="text-align: right;">28</td><td style="text-align: right;">26</td></tr>
<tr><td style="text-align: right;">36</td><td style="text-align: right;">3</td><td style="text-align: right;">18</td><td style="text-align: right;">33</td><td style="text-align: right;"> 9</td></tr>
</tbody>
</table></small></div>

## Optimizing Numerical Computation

We're going to go over some common techniques to speed up numerical computation. This will be very useful when optimizing code for speed.

### Actually Use Numpy

Instead of mixing Numpy and Python lists together, use Numpy for as much of the computation as possible. You make orders of magnitude of gain here!

This is because Numpy operations are (1) performed in low-level C code and don't incur python interpreter overhead and (2) vectorized.

Here's an example where we add 1,000,000 integers together:

```python
s = np.ones(1000000)

def loopsum(m):
    rv = 0
    for i in range(len(m)):
        rv += m[i]
    return rv

def npsum(m):
    return np.sum(m)
```

```python
%%timeit
loopsum(s) == 1000000
```

<pre>
202 ms ± 8.83 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

</pre>


```python
%%timeit
npsum(s) == 1000000
```

<pre>
342 µs ± 16.2 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)

</pre>


We get at least 500x speedup with Numpy.

### Rewrite Code To Consolidate Multiplication

As a general rule of thumb, here are the relative costs of different multiplications:

| Left | Right | Type | Cost ($n$ == length) | Remarks |
| ------ | ------ | ---- |:----:|:----- |
| vector | vector | elementwise | $\mathcal O (n)$ |
| vector | vector | dot         | $\mathcal O (n)$ |
| matrix | vector | elementwise | $\mathcal O (n^2)$ | *(both row/column broadcasting)*
| matrix | vector | matmul      | $\mathcal O (n^2)$ |
| matrix | matrix | elementwise | $\mathcal O (n^2)$ |
| matrix | matrix | matmul      | $\mathcal O (n^3)$ |

Additionally, constructing a vector takes $\mathcal O(n)$ and a matrix takes $\mathcal O(n^2)$.

You can improve the speed of your computation considerably by rewriting your algorithm to avoid more expensive operations; a good sign that you can do this is when you have multiple inputs that are larger than the size of your output.

#### Example 1

Lets work through an example. We wish to calculate $A \* B \* \vec v$, which are matrices/vectors of size 1000. We can compute this two ways:

 1. $A \* (B \* \vec v)$
 2. $(A \* B) \* \vec v$

Assign a cost to each operation and figure out the size of each intermediate state to figure out which operation will be quicker.

```python
A = np.random.randint(1000, size=(1000, 1000))
B = np.random.randint(1000, size=(1000, 1000))
v = np.random.randint(1000, size=(1000,))
```

```python
%%timeit
(A @ B) @ v
```

<pre>
1.23 s ± 88.6 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

</pre>


```python
%%timeit
A @ (B @ v)
```

<pre>
1.44 ms ± 3.81 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)

</pre>


We also check that they give the same answer:

```python
all((A @ B) @ v == A @ (B @ v))
```

We can see why method 1 is about a thousand times faster than method 2 by breaking down the operations:

  1. In method 1, we first evaluate the matrix-vector product $B \* \vec v$ which incurs $\mathcal O(n^2)$ time and produces a vector of length $n$. The second operation is also a matrix-vector product which takes $\mathcal O(n^2)$ time:
  $$
  \underbrace{A * \underbrace{(B * \vec v)}_{\mathcal O(n^2)}}_{\mathcal O(n^2)}
  $$
  The overall time is $\mathcal O(n^2)$
  2. In method 2, we first evaluate the matrix-matrix product $(A \* B)$, which takes $\mathcal O(n^3)$ time. The subsequent matrix-vector takes $\mathcal O(n^2)$ time.
  $$
  \underbrace{\underbrace{(A * B)}_{\mathcal O(n^3)} * \vec v)}_{\mathcal O(n^2)}
  $$
  The overall operation is dominated by the $\mathcal O(n^3)$ time.

This suggests that method 2 should be slower by a factor of $n$, which is 1,000 in our example.

#### Example 2

Let's try this again, this time a more subtle use-case.

We wish to compute $A \* \text{diag}(\vec u) \* \vec v$, where $\text{diag}(\vec u)$ is an otherwise zero matrix with the diagonals set to $\vec u$. There are two ways to do this that we have already discussed:

  1. $(A \* \text{diag}(\vec u)) \* \vec v$ which takes $\mathcal O(n^3)$ time
  2. $A \* (\text{diag}(\vec u) \* \vec v)$ which takes $\mathcal O(n^2)$ time

We observe that $\text{diag}(\vec u) \* \vec v$ is the same as taking the elementwise product $\vec u \circ \vec v$. We use this to rewrite the operation:

  3. $A \* (\vec u \circ \vec v)$ which takes $\mathcal O(n^2)$ time

Even though (2) and (3) are the same asymptotically, notice that (2) takes _three_ $\mathcal O(n^2)$ operations but (3) only takes one. Question: _why three operations?_

At a moderate $n = 1000$, lets see which is quicker:

```python
A = np.random.randint(1000, size=(1000, 1000))
u = np.random.randint(1000, size=(1000,))
v = np.random.randint(1000, size=(1000,))
```

```python
%%timeit
# Method 2
A @ (np.diag(u) @ v)
```

<pre>
1.77 ms ± 13.1 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)

</pre>


```python
%%timeit
# Method 3
A @ (u * v)
```

<pre>
723 µs ± 10.8 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)

</pre>


By saving the extra $\mathcal O(n^2)$ operations, we speed our computation up considerably. We check that the output is the same:

```python
all(A @ (np.diag(u) @ v) == A @ (u * v))
```

Great!
