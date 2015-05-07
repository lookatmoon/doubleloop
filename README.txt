
Steps
=====
1. Data Set level
1) make the data file: docID [TAB] content
2) Index the data; build a searcher
3) make the label file: docID [TAB] label
4) make the term ID file: term termID termIDF

2. Task specific
1) make the eval data file, +1/-1 in proportion
2) make seed, avoid the eval data
3) run: avoid adding eval data or label: restricting IDs.



