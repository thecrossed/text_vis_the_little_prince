Text Visualization of The Little Prince

Reading novels is a fascinating but long journey. An enormous amount of text information sometimes gets readers lost. However, text visualization can take them out of the woods by revealing text content at either document level or word level.[1]  In this article, I used techniques as Summarizing a Single Document, PhraseNet, and Visualizing Sentiments and Emotions[2] to visualize my favorite novella The Little Prince.


1. Showing The Little Prince at the Word Level

TagCloud is one of the most intuitive and commonly used techniques for visualizing words.[3] I used WordCloud, an open source Python library to generate high-frequency words in The Little Prince. In order to create a shape for this TagCloud, I found a PNG file depicting the figure of the main character, little prince, to become the mask. 


2. Summarizing The Little Prince at the sentence level

The graph illustrates two aspects of The Little Prince: the number of sentences in each chapter and the length of each sentence. To clearly show these two kinds of information, I employed two indicators as length and color respectively - the longer the horizontal bar, the more sentences in a chapter; the darker a single cell, the more words in a sentence represented.


3. PhraseNet of The Little Prince

PhraseNet employs a node-link diagram, in which graph nodes are keywords and links represent relationships among keywords.[4] I adopted Natural Language ToolKit(NLTK)  to tokenize each sentence in The Little Prince and found the ten most frequent nouns in this novel. Next, I emphasized these nouns: prince, planet, flower, sheep, stars, king, fox, time, grown-ups, geographer by painting them with blue circles, more frequently shown ones surrounded with larger circles. Besides, I deployed each other noun which appeared together with any of the ten top nouns in the same sentence, using orange to plot the nouns(nodes) which appeared more than once and gray for the else.

4. Visualizing Sentiments and Emotions of The Little Prince

Illustrating the change of sentiments in sentences over chapters helps readers get a general sense of the sentiment trend in the whole text. Take the picture as an example, I employed TextBlob, a python library for text analysis, to calculate the sentiment score of each sentence and plotted them in a 2D diagram. The vertical axis displays the sentiment score, and the horizontal axis shows the sequence of all the sentences in The Little Prince. What is more interesting is I colored all the dots in orange which represent sentences with the word rose inside. Observers can easily find which part of the text where the word rose most occurs and the general sentiment score on this word (mostly above average, positive).




References
1. Cao, Nan, Cui, Weiwei: Overview of Text Visualization Techniques. Introduction to Text Visualization, 11 (2016)
2. Cao, Nan, Cui, Weiwei: Overview of Text Visualization Techniques. Introduction to Text Visualization, 16-19 (2016)
3. Kaser, O., Lemire, D.: Tag-cloud drawing: algorithms for cloud visualization. arXiv preprint
cs/0703109 (2007)
4. Van Ham, F., Wattenberg, M., Viégas, F.B.: Mapping text with phrase nets. IEEE Trans. Vis.
Comput. Graph. 15(6), 1169–1176 (2009)





