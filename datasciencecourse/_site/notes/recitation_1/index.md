# Recitation 1

### Homework Tips

* start early; you should already be mostly through `hw1_scraper`
* once the environment is set up, `hw1_get_started` should take you ~10 minutes to finish
* 20 minutes reading documentation will save you 8 hours of writing code
* write your own tests! The tests we give you clarify what the code should output, not (sufficiently) show that your code is correct.
* The last TA hours before an assignment due is always the most crowded. Don't rely on those to complete your assignment.
* Talk to other students taking the course -- they can help you and you can help them.
* Look for the "Common Problems in Homework x" post on Diderot before asking questions online.

### Autograder

* The autograder status is announced on Diderot: if it is down, the announcement is pinned and I provide updates every few hours.
* Autograder changes are also announced and (briefly) pinned. I will announce changes in some detail. I will likely _not_ regrade assignments that have received full scores.
* Currently we have no submission limit. Compute power is a finite resource and ~200 students share it; if someone abuses it we will limit the number of submissions.
* You should be able to see the Autograder output. Use the output to guide your debugging.


### TA Hours

* Come early, come often. You learn a lot by listening to others problems and discussing your problems with them
* We triage problems, group students, and deal with groups. For large classes, this is vastly more efficient.
  * We will correct misunderstandings, but will not directly give you the correct answer
  * TAs may direct you to other students with similar problems and ask you to work together until we can see you
  * Students are _not_ seen in a first-come-first-serve queue

### Environment Setup

Python environments take a little practice to get exactly right. It is very easy to spectacularly mess it up, as Randall Munroe illustrates on xkcd:

{% include image.html img="https://imgs.xkcd.com/comics/python_environment.png" caption=""%}

We encourage you to use Vagrant because it sets up a clean, repeatable environment in a virtual machine. You may also wish to use Anaconda (a popular all-in-one solution), or Windows Subsystem for Linux (run Ubuntu/Fedora/Debian as a Windows app).

Also, **never run `sudo pip install ...`**. Instead, try `pip --user install ...`

{% include image.html img="https://i.imgur.com/T1fubRO.png" caption=""%}

Running `sudo pip install ...` gives random people on the internet, and me, root access to your machine. That is a Very Bad Idea.

## Writing Tests

Writing good tests requires you understand the problem domain and _what can go wrong_. Typically, you learn what can go wrong by making those mistakes over and over.

### Approach
When writing your tests, there are two approaches:

- bottom-up/unit testing, where you test the smallest part of your code for correctness and gradually move up the function hierarchy. These are easy to write at low levels but get expensive as the output becomes more complex.
- top-down/global/functional/integration testing, where you have some input (ideally real-world input) and manually verified output. These are (generally) expensive to build, though you can trade quality for time by writing synthetic tests.

We give you both tests, and our grader runs both types of tests on your code, but we generally weigh top-down tests much more. If you cannot pass a test, it is likely that there is an [unknown unknown](https://en.wikipedia.org/wiki/Johari_window) factor at play and you should try your code on more tests.

### Complexity

Also, you don't write a single test so much as a suite of tests. Instead of just writing many large tests, begin by writing the smallest possible test to check a feature then copy it and add complexity to it. If you discover a bug, the tests that pass give you valuable information on what must be causing it.

### Regression

Write _regression_ tests. When you discover a bug

1. write a test that should fail
2. check that the test does fail
3. fix the bug
4. check that the test now passes

This way you will catch similar bugs in the future.

For `hw1_xml_parser`, we have a thread where you can share your tests; regression tests are especially valuable because if you made a mistake it is likely that others will as well.

## Scraping with `requests`

Read the [documentation](http://docs.python-requests.org/en/master/).

```python
import requests
```

### What is HTTP?

A _protocol_ for automatically requesting (and getting) data from text-based servers. The using human-readable text

When you navigate to [www.example.com](www.example.com), here's what my browser sends as plain (UTF-8) text:

```
GET / HTTP/1.1
Host: www.example.com
User-Agent: Mozilla/5.0 (X11; Linux x86_64; rv:68.0) Gecko/20100101 Firefox/68.0
Accept: text/html
Accept-Language: en-US,en;q=0.5
Accept-Encoding: gzip, deflate
```

We begin with the request type (`GET`), the path (`/`), and the protocol version (`HTTP/1.1`). The browser then identifies itself and advertises its capabilities.

The server sends back:

```
HTTP/1.1 200 OK
Content-Encoding: gzip
Content-Type: text/html; charset=UTF-8
Last-Modified: Fri, 09 Aug 2013 23:54:35 GMT
Server: ECS (phd/FD6D)
Content-Length: 606

<!doctype html>
<html>
...
```

The response begins with the headers: the protocol version `HTTP/1.1` and that the request succeeded (`200 OK`). Then it gives you information about the response and the server. Finally, it leaves a blank line and then gives you the _response body_.

`requests` is a library that handles the heavy lifting of this process. It

- constructs request headers, including handling encoding
- establishes a conection
- transfers the request (and respects the fiddly little protocol details)
- receives the response
- exposes the response headers and body to the programmer

### Using Requests

```python
response = requests.get("http://www.google.com/search",
                        params={ "query": "python metaclass", "source":"chrome" })
```

We send request headers that look something like this:
```
GET /?query=python%20metaclass&source=chrome HTTP/1.1
Host: www.google.com
Accept-Encoding: gzip, deflate, compress
Accept: */*
User-Agent: python-requests/2.1.0 CPython/3.6.7 Linux/3.2.0-23-generic-pae
```

Note how the GET parameters are added to the requested filename, and they are encoded to remove some characters. `requests` handles this for you automatically.

Lets examine the response we get. We can get the response url and the [status code](https://en.wikipedia.org/wiki/List_of_HTTP_status_codes) as properties:

```python
response.url
```

```python
response.status_code # `200` means the request was successful
```

```python
from pprint import pprint # Pretty-printer
pprint(dict(response.headers))
```

<pre>
{&#x27;Cache-Control&#x27;: &#x27;private, max-age=0&#x27;,
 &#x27;Content-Encoding&#x27;: &#x27;gzip&#x27;,
 &#x27;Content-Type&#x27;: &#x27;text/html; charset=ISO-8859-1&#x27;,
 &#x27;Date&#x27;: &#x27;Thu, 05 Sep 2019 03:58:08 GMT&#x27;,
 &#x27;Expires&#x27;: &#x27;-1&#x27;,
 &#x27;P3P&#x27;: &#x27;CP=&quot;This is not a P3P policy! See g.co/p3phelp for more info.&quot;&#x27;,
 &#x27;Server&#x27;: &#x27;gws&#x27;,
 &#x27;Set-Cookie&#x27;: &#x27;1P_JAR=2019-09-05-03; expires=Sat, 05-Oct-2019 03:58:08 GMT; &#x27;
               &#x27;path=/; domain=.google.com; SameSite=none, CGIC=IgMqLyo; &#x27;
               &#x27;expires=Tue, 03-Mar-2020 03:58:08 GMT; path=/complete/search; &#x27;
               &#x27;domain=.google.com; HttpOnly, CGIC=IgMqLyo; expires=Tue, &#x27;
               &#x27;03-Mar-2020 03:58:08 GMT; path=/search; domain=.google.com; &#x27;
               &#x27;HttpOnly, &#x27;
               &#x27;NID=188=kT-c41uxIh8UsAuyXVBzg6CukUgfuwceNA2AyFvaosgsM_B0X9lps4MwweFhF2VwzbEOoX_hD6e4tHRKFfuKF-dNO0u0Rl1RUiUCILdSw9aQvnWdOXPlKg0ZeG-sfiMkPwX3YKhgx7XMTXMBpye1rZgNWkcbCA2ZKjEAmYjpoOQ; &#x27;
               &#x27;expires=Fri, 06-Mar-2020 03:58:08 GMT; path=/; &#x27;
               &#x27;domain=.google.com; HttpOnly&#x27;,
 &#x27;Transfer-Encoding&#x27;: &#x27;chunked&#x27;,
 &#x27;X-Frame-Options&#x27;: &#x27;SAMEORIGIN&#x27;,
 &#x27;X-XSS-Protection&#x27;: &#x27;0&#x27;}

</pre>


### Response body and encoding.

```python
(type(response.text), response.text[:100])
```

```python
(type(response.content), response.content[:100])
```

Notice that the returned type is different between `.text` and `.content`.

- `.content` returns a `bytes` object that represents binary data.
- `.text` returns a `str` object which contains characters, decoded from `.content` using the `encoding` field from the `'Content-Type'` header.

#### Content-type conflicts

Notice there's a conflict in the `Content-Type` header and in the HTML itself:

```python
response.headers["Content-Type"]
```

```python
response.text[37:59]
```

This happens very, very often. BeautifulSoup4 and many other libraries accept the bytes directly and automatically figure out the encoding. (This was the autograder bug.) Here's what happens when you use the wrong encoding:

```python
print("Bytes:")
print(b'\xf0\x9f\x92\xa9')
print("Using the Content-Type encoding: [ISO-8859-1]")
print(b'\xf0\x9f\x92\xa9'.decode("ISO-8859-1"))
print("Using the <meta> tag encoding: [UTF-8]")
print(b'\xf0\x9f\x92\xa9'.decode("UTF-8"))

```

<pre>
Bytes:
b&#x27;\xf0\x9f\x92\xa9&#x27;
Using the Content-Type encoding: [ISO-8859-1]
ð©
Using the &lt;meta&gt; tag encoding: [UTF-8]
💩

</pre>


## Regular Expressions

Before you use regular expressions, [read this](https://blog.codinghorror.com/regular-expressions-now-you-have-two-problems/). If you're interested in practicing with RegExes, you can [read about Regex Golf](https://nbviewer.jupyter.org/url/norvig.com/ipython/xkcd1313.ipynb) and then [try it yourself](http://alf.nu/RegexGolf).

Regular expressions are a way to find or extract text from strings. For this tutorial, you should keep open one of these cheat sheets: [MIT](http://web.mit.edu/hackl/www/lab/turkshop/slides/regex-cheatsheet.pdf), [RegexLib](http://regexlib.com/cheatsheet.aspx) and an online RegEx tester ([recommended](https://regex101.com/)).

There are many flavors of RegExps; they handle the basics the same way but have subtle differences around backreferences.

We'll set up some a testing function and use it to run `re.match(...)` on some regular expressions and examples.

```python
import re
from IPython.display import display, Markdown, Latex

# FEEL FREE TO IGNORE THIS CODE

def _match(regex, example, search=re.search):
    m = search(regex, example)
    if m:
        st, en = m.span()
        return f"{example[:st]}<u>{example[st:en]}</u>{example[en:]}"
    else:
        return f"<s>{example}</s>"

def pm(examples, regexes, search=re.search):
    examples = examples if isinstance(examples, list) else [examples]
    regexes = regexes if isinstance(regexes, list) else [regexes]
    output=[]
    for regex in regexes:
        display(Markdown(f"**re.{search.__name__} {regex} :** " + ", ".join(_match(regex, example, search) for example in examples)))
        
def pmg(examples, regex, match=re.match):
    examples = examples if isinstance(examples, list) else [examples]
    for example in examples:
        a = match(regex, example)
        display(Markdown(f"**re.{match.__name__}(..., {example}) :** " + ", ".join(f"'{m}'" for i, m in enumerate(a.groups()))))
```

Lets begin with a warmup. Here's how you match one character:

```python
pm(["bat", "bit", "bot", "but", "batty", "bitty", "and", "or", "not"],
   "b[aeiou]t")
```

And here's how you match distinct options. Note how the `a` in `batty` is matched -- most RegExp engines will find the first match.

```python
pm(["bat", "bit", "bot", "but", "batty", "bitty", "and", "or", "not"],
   ["(tt|a)", "(tt|a)"])
```

`re.search` starts from any position; you can use `re.match` to check the string starting from a position. (There are other options, like `re.findall`)

```python
pm(["bot", "talbot", "botobot"], ["bot"], re.search)
pm(["bot", "talbot", "botobot"], ["bot"], re.match)
pm(["bot", "talbot", "botobot"], ["bot"], re.fullmatch)
```

Matching some occurences can be controlled using this syntax:

* To match `a` exactly once, use `a`
* To match `a` zero or one times, use `a?`
* To match `a` zero or more times, use `a*`
* To match `a` one or more times, use `a+`
* To match `a` exactly `n` times, use `a{n}`.
* To match `a` between `n` and `m` times, use `a{n,m}`.
* To match `a` at least `n`, use `a{n,}`.
* To match `a` at most `m`, use `a{,m}`.

```python
pm(["bt", "bat", "baat", "baaat", "baaaat"],
   ["bat", "ba?t", "ba+t", "ba*t", "ba{2,3}t"])
```

There is plenty more syntax:

* Any character (except special characters, `".$*+?{}\[]|() `), just matches itself
* Putting a group of characters within brackets `[abc]` will match any of the characters a, b, or c
* Putting a caret within the bracket matches anything but these characters, i.e., `[^abc]` matches any character except a, b, or c.
* The special character `\d` will match any digit, i.e. `[0-9]`
* The special character `\w` will match any alphanumeric character plus the underscore; i.e., it is equivalent to `[a-zA-Z0-9_]`.
* The special character `\s` will match whitespace, any of `[ \t\n\r\f\v]` (a space, tab, and various newline characters)
* The special character `.` (the period) matches any character. In their original versions, regular expressions were often applies line-by-line to a file, so by default `.` will not match the newline character. If you want it to match newlines, you pass `re.DOTALL` to the “flags” argument of the various regular expression calls.


There are more details; consult the [documentation](https://docs.python.org/3/library/re.html). Also, note that the Python 3 `re` module does not guarantee Unicode support.

### Capture Groups

This is how we extract parts of structured data:

```python
pmg(["10:00 am", "10:20 am", "3:11 pm"],
    r"((\d\d?):(\d\d)) (am|pm)")
```

### Greedy Matching

RegExes' multi-selectors (`*+`) match _the longest possible string_ by default. For example:

```python
pm(["(1 + 2) * (3 + 4)"],
   [r"\(.*\)", "\(.+\)", "\(.{2,}\)"])
```

You can make them match the shortest possible string instead using `?`. This is called _lazy matching_.

```python
pm(["(1 + 2) * (3 + 4)"],
   [r"\(.*?\)", "\(.+?\)", "\(.{2,}?\)"])
```
