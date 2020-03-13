# Recitation 5

## Homework Tips

* Be familiar with the [What Can I Ask On Diderot?](www.diderot.one/course/10/dosts/?is_inbox=yes&dost=4658) policy
* Talk to other students taking the course -- they can help you and you can help them.
* Look for the "Common Problems in Homework x" post on Diderot before asking questions online.
* We limit the number of submissions in parts of this homework.

```python
import numpy as np
import scipy.sparse as sp
from IPython.display import HTML, display
import tabulate

def pp(a):
    display(HTML(tabulate.tabulate(a, tablefmt='html')))
    

def aa(a, v):
    print(f"{'CORRECT' if np.allclose(a, v) else 'INCORRECT'}\t{a}")
```

## Working With Small Numbers

In parts of this course you have had to deal with small probabilities, likelihoods, and scores. Floating-point numbers have limited precision, and when your numbers become very small, it can cause problems. For example:

```python
x = np.array([2**-150, 2**-151], dtype=np.float32)
x[0]/(x[0] + x[1]) 
```

### More Precision?

You can escape this using higher-precision floating-point numbers, but that's slower, takes more memory, and doesn't fundamentally solve the problem.

```python
x = np.array([2**-150, 2**-151], dtype=np.float64)
aa(x[0]/(x[0] + x[1]), 2/3)
```

<pre>
CORRECT	0.6666666666666666

</pre>


```python
x = np.array([2**-1490, 2**-1491], dtype=np.float64)
aa(x[0]/(x[0] + x[1]), 2/3)
```

<pre>
/home/gauravmm/.pyenv/versions/3.6.7/envs/homework1/lib/python3.6/site-packages/ipykernel_launcher.py:2: RuntimeWarning: invalid value encountered in double_scalars
  

</pre>


Numpy will tell you the precision it can give you:

```python
finfo = np.finfo(np.float32)
print("Number:")
pp([(k, finfo.__dict__[k]) for k in ["bits", "eps", "precision"]])
print("Exponent:")
pp([(k, finfo.__dict__[k]) for k in ["iexp", "minexp", "maxexp"]])
```

<div><small><table>
<tbody>
<tr><td>iexp  </td><td style="text-align: right;">   8</td></tr>
<tr><td>minexp</td><td style="text-align: right;">-126</td></tr>
<tr><td>maxexp</td><td style="text-align: right;"> 128</td></tr>
</tbody>
</table></small></div>

This is how the number is represented:

{% include image.html img="https://upload.wikimedia.org/wikipedia/commons/d/d2/Float_example.svg" caption="From Wikipedia: Single-precision floating-point format"%}

Single-precision floating-point format, from [Wikipedia](https://en.wikipedia.org/wiki/Single-precision_floating-point_format#/media/File:Float_example.svg), used under CC-BY-SA.

That's why we ask you to work with the _logarithm_ of the probability density; we are trading off precision in the fraction/mantissa for additional range of values that we can represent.

```python
x = np.array([-1490 * np.log(2), -1491 * np.log(2)])
aa(x[0]/(x[0] + x[1]), 2/3)
```

<pre>
INCORRECT	0.4998322710499832

</pre>


This doesn't work, because you _can't directly add logarithms_. So far, in our language models and Bayes homework, we've only asked you to multiply numbers, which you do by adding their logarithms.

### One Weird Trick

For Unsupervised Learning (and in the future), you'll need to add very small numbers together when using them in a fraction. The trick to doing that is to divide them by a small number.

If you just use the numerator:
$$
\frac{e^{-x}}{e^{-x} + e^{-y}}
= \frac{e^{-x}}{e^{-x} + e^{-y}} * \frac{e^{x}}{e^{x}}
= \frac{e^{0}}{e^{0} + e^{x-y}}
= \frac{1}{1 + e^{x-y}}
$$

This works if the exponents $x$ and $y$ are similar orders-of-magnitude. A good trick to use is to pick the smaller of $x$ and $y$ (that is, the number closer to 1 between $e^{-x}$, $e^{-y}$).

_Why is this heuristic good?_

Implementing this:

```python
x = np.array([-1490 * np.log(2), -1491 * np.log(2)])

x -= x.max() # This is the trick
x = np.exp(x) # Now we can convert it from the log-value to the value.

aa(x[0] / (x[0] + x[1]), 2/3)
```

<pre>
CORRECT	0.6666666666666544

</pre>


Great!

## Neural Networks.

{% include image.html img="https://upload.wikimedia.org/wikipedia/commons/4/46/Colored_neural_network.svg" caption=""%}
Colored Neural Network, from [Wikimedia Commons](https://en.wikipedia.org/wiki/File:Colored_neural_network.svg), used under CC-BY-SA.

The basic layer type is a fully-connected layer, shown here. Our data has $64\*64\*3 = 12288$ elements; If we had one fully-connected hidden layer with a modest 1000 elements, we would need a total of:

$$
(12,288 * 1,000 + 1,000) + (1,000 * 1 + 1) = 12,290,001~\text{elements}
$$

The problem with fully-connected networks was discussed by Prof. Kolter in class ([this slide](www.datasciencecourse.org/slides/deep_learning.pdf#page=26)).

### Hierarchical Structure

Visual processing systems in nature are organized hierarchically [(Hubel & Wiesel, 1959)](https://www.nobelprize.org/prizes/medicine/1981/summary/). In particular, the building blocks are _similar_ and _local_. That's where convolutional layers come in:

{% include image.html img="https://d2l.ai/_images/correlation.svg" caption=""%}
Convolutional Layer from [Dive Into Deep Learning](https://d2l.ai/chapter_convolutional-neural-networks/padding-and-strides.html), used under CC-NC-BY-SA.

We slide the same small matrix over the entire input space, performing the _convolution_ at each position and storing the output in the corresponding position. This is the building block of almost every network operating on image data, and the associated operation is `nn.Conv2d`.

Given an input vector shaped like `[example, channel, height, width]`, it slides over the `height` and `width` dimensions and convolves any number of input channels to a (potentially different) number of output channels. You should experiment with these options:

```python
torch.nn.Conv2d(
    in_channels, out_channels,
    kernel_size,
    stride=1,
    padding_mode='zeros')
```

Note that the output size is different than the input size. Experiment with `kernel_size` and `padding_mode` and see how they change the shape of the output. (That's why we print the shapes in the test case!)

The sliding-window convolution is a popular building block, but you can perform other operations, like taking the _maximum_ value in each channel. These are called [pooling layers](https://pytorch.org/docs/stable/nn.html#pooling-layers).

Instead of pooling, you can also include non-linear operations like `nn.ReLU`, `nn.Tanh`, etc. These are sometimes called [_activation functions_](https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity).

### Tips

  - Look at the network linked to in the handout; the typical structure is to use a few convolutional and/or pooling layers to get the size manageable and then use fully connected layers.
  - Convolutional layers are quite small, you can afford to stack them. We Have To Go Deeper!
  - You can use half (or more) of your parameters in the fully-connected layers; that's normal.
  - Remember your non-linear operation between linear operations like fully-connected layers. (Why?)
  - A useful thing to think about is the _receptive field_ of each cell in your intermediate tensor. Here's an [online calculator](https://fomoro.com/research/article/receptive-field-calculator).

If you want a more detailed introduction, [this open-source book](https://d2l.ai/) is pretty good.
