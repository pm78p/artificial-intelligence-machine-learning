---
__Files in Branch:__

- __[Machine learning](https://github.com/pm78p/root/tree/main/Machine%20learning)__ - Here, there are number of projects which are related to ML field.
- __[computer vision](https://github.com/pm78p/root/tree/main/computer%20vision)__ - Projects are related to Computer Vision field.
- __[deep learning](https://github.com/pm78p/root/tree/main/deep%20learning)__ - Projects are related to DL field.
- __[kaggle competion](https://github.com/pm78p/root/tree/main/kaggle%20competion)__ - Project is related to Kaggle competition.
  

### [Machine learning](https://github.com/pm78p/root/tree/main/Machine%20learning)


[1-ICU italy dataset](https://github.com/pm78p/root/tree/main/Machine%20learning/1-ICU%20italy%20dataset) - Here we modeled ICU italy dataset with help of the some methods
(Multy layer perceptron, Random forest, ada boost, linear regression). The results for each methods listed below:

![results](https://github.com/pm78p/root/blob/main/1-icu_result.png)

*** [Code](https://github.com/pm78p/root/blob/main/Machine%20learning/1-ICU%20italy%20dataset/codes.ipynb)

[2-sarcasm detection dataset](https://github.com/pm78p/root/tree/main/Machine%20learning/2-sarcasm%20detection%20dataset) - Here After doing some Data visualization and  Data
preprocessing we modeled sarcasm on reddit dataset from kaggle with help of the some methods.
In one way we use TF-IDF embedding method, then we train, a. Logistic Regression - b. SVM - c. Ada Boost, models on data. 
In other way we embedd text with the help of the Word2Vec and train RNN model.
The results for each methods listed below:

![results](https://github.com/pm78p/root/blob/main/2-sarcasm_result.png)

*** [Code](https://github.com/pm78p/root/blob/main/Machine%20learning/2-sarcasm%20detection%20dataset/codes.ipynb)

[3-data cleaning & EDA & Dimension Reduction](https://github.com/pm78p/root/tree/main/Machine%20learning/3-data%20cleaning%20%26%20EDA%20%26%20Dimension%20Reduction) -

*** [Code](https://github.com/pm78p/root/blob/main/Machine%20learning/3-data%20cleaning%20%26%20EDA%20%26%20Dimension%20Reduction/codes.ipynb)

[4-regression](https://github.com/pm78p/root/tree/main/Machine%20learning/4-regression) - Here we use "The Boston Housing" Dataset. Fisrt, we compare gradient descent and stochastic gradient descent on this dataset. Next, a. implement 1st order regression with SSE
as cost function - b. 3rd order regression with SSE as cost function - c. 3rd order regression with SSE as cost function and a regularization term and predict test data.
Last, we use a 10-fold cross validation.

*** [Code](https://github.com/pm78p/root/blob/main/Machine%20learning/4-regression/codes.ipynb)

[5-logistic reg & ada boost](https://github.com/pm78p/root/tree/main/Machine%20learning/5-logistic%20reg%20%26%20ada%20boost) - Here we use "Blood Transfusion Service Center"
dataset. First, we train logistic regression classiﬁer and a decision tree to classify the dataset. Then, fine tuning models. Finally, Train an AdaBoost classiﬁer that each weak 
learner is a stump (Stumps are decision trees with depth one).

*** [Code](https://github.com/pm78p/root/blob/main/Machine%20learning/5-logistic%20reg%20%26%20ada%20boost/codes.ipynb)

[6-probabilistic graphical models](https://github.com/pm78p/root/tree/main/Machine%20learning/6-probabilistic%20graphical%20models) - Here we try to use probabilistic graphical models
to denoising messages in a related dataset

*** [Code](https://github.com/pm78p/root/blob/main/Machine%20learning/6-probabilistic%20graphical%20models/codes.ipynb)

[7-clustering k-means & EM & GMM](https://github.com/pm78p/root/tree/main/Machine%20learning/7-clustering%20k-means%20%26%20EM%20%26%20GMM) - Here we made a random dataset via sklearn 
by make_classification function in this library. Then, implementing k-means and EM clusterings to classify the datas.

*** [Code](https://github.com/pm78p/root/blob/main/Machine%20learning/7-clustering%20k-means%20%26%20EM%20%26%20GMM/codes.ipynb)

[8-CNN](https://github.com/pm78p/root/tree/main/Machine%20learning/8-CNN) - 

*** [Code](https://github.com/pm78p/root/blob/main/Machine%20learning/8-CNN/codes.ipynb)





:   Definition 1
with lazy continuation.

Term 2 with *inline markup*

:   Definition 2

        { some code, part of Definition 2 }

    Third paragraph of definition 2.


---

# h1 Heading 8-)
## h2 Heading
### h3 Heading
#### h4 Heading
##### h5 Heading
###### h6 Heading


## Horizontal Rules

___

---

***


## Typographic replacements

Enable typographer option to see result.

(c) (C) (r) (R) (tm) (TM) (p) (P) +-

test.. test... test..... test?..... test!....

!!!!!! ???? ,,  -- ---

"Smartypants, double quotes" and 'single quotes'


## Emphasis

**This is bold text**

__This is bold text__

*This is italic text*

_This is italic text_

~~Strikethrough~~


## Blockquotes


> Blockquotes can also be nested...
>> ...by using additional greater-than signs right next to each other...
> > > ...or with spaces between arrows.


## Lists

Unordered

+ Create a list by starting a line with `+`, `-`, or `*`
+ Sub-lists are made by indenting 2 spaces:
  - Marker character change forces new list start:
    * Ac tristique libero volutpat at
    + Facilisis in pretium nisl aliquet
    - Nulla volutpat aliquam velit
+ Very easy!

Ordered

1. Lorem ipsum dolor sit amet
2. Consectetur adipiscing elit
3. Integer molestie lorem at massa


1. You can use sequential numbers...
1. ...or keep all the numbers as `1.`

Start numbering with offset:

57. foo
1. bar


## Code

Inline `code`

Indented code

    // Some comments
    line 1 of code
    line 2 of code
    line 3 of code


Block code "fences"

```
Sample text here...
```

Syntax highlighting

``` js
var foo = function (bar) {
  return bar++;
};

console.log(foo(5));
```

## Tables

| Option | Description |
| ------ | ----------- |
| data   | path to data files to supply the data that will be passed into templates. |
| engine | engine to be used for processing templates. Handlebars is the default. |
| ext    | extension to be used for dest files. |

Right aligned columns

| Option | Description |
| ------:| -----------:|
| data   | path to data files to supply the data that will be passed into templates. |
| engine | engine to be used for processing templates. Handlebars is the default. |
| ext    | extension to be used for dest files. |


## Links

[link text](http://dev.nodeca.com)

[link with title](http://nodeca.github.io/pica/demo/ "title text!")

Autoconverted link https://github.com/nodeca/pica (enable linkify to see)


## Images

![Minion](https://octodex.github.com/images/minion.png)
![Stormtroopocat](https://octodex.github.com/images/stormtroopocat.jpg "The Stormtroopocat")

Like links, Images also have a footnote style syntax

![Alt text][id]

With a reference later in the document defining the URL location:

[id]: https://octodex.github.com/images/dojocat.jpg  "The Dojocat"


## Plugins

The killer feature of `markdown-it` is very effective support of
[syntax plugins](https://www.npmjs.org/browse/keyword/markdown-it-plugin).


### [Emojies](https://github.com/markdown-it/markdown-it-emoji)

> Classic markup: :wink: :crush: :cry: :tear: :laughing: :yum:
>
> Shortcuts (emoticons): :-) :-( 8-) ;)

see [how to change output](https://github.com/markdown-it/markdown-it-emoji#change-output) with twemoji.


### [Subscript](https://github.com/markdown-it/markdown-it-sub) / [Superscript](https://github.com/markdown-it/markdown-it-sup)

- 19^th^
- H~2~O


### [\<ins>](https://github.com/markdown-it/markdown-it-ins)

++Inserted text++


### [\<mark>](https://github.com/markdown-it/markdown-it-mark)

==Marked text==


### [Footnotes](https://github.com/markdown-it/markdown-it-footnote)

Footnote 1 link[^first].

Footnote 2 link[^second].

Inline footnote^[Text of inline footnote] definition.

Duplicated footnote reference[^second].

[^first]: Footnote **can have markup**

    and multiple paragraphs.

[^second]: Footnote text.


### [Definition lists](https://github.com/markdown-it/markdown-it-deflist)

Term 1

:   Definition 1
with lazy continuation.

Term 2 with *inline markup*

:   Definition 2

        { some code, part of Definition 2 }

    Third paragraph of definition 2.

_Compact style:_

Term 1
  ~ Definition 1

Term 2
  ~ Definition 2a
  ~ Definition 2b


### [Abbreviations](https://github.com/markdown-it/markdown-it-abbr)

This is HTML abbreviation example.

It converts "HTML", but keep intact partial entries like "xxxHTMLyyy" and so on.

*[HTML]: Hyper Text Markup Language

### [Custom containers](https://github.com/markdown-it/markdown-it-container)

::: warning
*here be dragons*
:::
