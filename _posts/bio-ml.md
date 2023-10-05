---
layout: post
title:  Applying AI to Biological Data: Challenges and Lessons
date:   2023-10-05
description: Lessons and challenges from training AI on biological datasets.
tags: AI, Biology
---

My work in the past few years has focused on translation of methods underlying
the remarkable advances in AI and NLP to biological sequence design problems.

Many of the most successful algorithms in NLP can be translated with little
modification to operate on biological sequences rather than natural
language. Take for example the extremely prolific Evolutionary Scale Model (ESM)
which is basically just the BERT masked-token prediction task applied to protein
sequences.

However, this line of work comes with unique challenges that include 1)


Imagine how much harder it would have been to train the GPT line of models if
individual reseachers performing experiments did not have the ability to prompt
one of the models and visually inspect its generated natural language?

Or rather, imagine that they could be they could only look at a few hundred
model outputs every month.

Eric Lander in 2004 famously said about the Human Genome Project
"Genome. Bought the book. Hard to read."

Translating the remarkable advances made in artificial intelligence and NLP from
the past few years towards biological applications has been a major focus of my
techical work for the past few years.

In this post, I'll give a picture of the experience and challenges of training
transformer language models on large biologicla sequence assets.

## Looking at the Data

It has been said that great ML researchers spend a surprising amount of time
looking at raw data. One particularly challenging aspect of working with
biological sequence data is that you cannot comprehend individual sequences in
the same way that

This means that as a reseracher, your interaction with the data is primarily
mediated by algorithms and software such as sequence alignment and database
search tools. Of course, you can always

is mediated by algorithms and tools for analyzing biological
data.

I find this both frustrating and

It means that in order to be

```
CGGAATTCAAGAAGCCCGAGGTGCATGTCGAGGTGCGGTTTGCCTCGTAAAAAAGCCGCA
ATTTAAAGTAATCGCAAACGACGATAACTACTCTCTAGCAGCTTAGGCTGGCTAGCGCTC
CTTCCATGTATTCTTGTGGACTGG_TTTTGGAGTGTCACCCTAACACCTGATCGCGACGG
AAACCCTGGCCGGGGTTGAAGCGTTAAAACTAAGCGGCCTCGCCTTTATCTACCGTGTTT
GTCCGGGATTTAAAGGTTAATTAAATGACAATACTAAACATGTAGTACCGACGGTCGAGG
CTTTTCGGACGGGG
```
A [transfer-messenger RNA](https://en.wikipedia.org/wiki/Transfer-messenger_RNA)
sequence from [RNAcentral](https://rnacentral.org) v22, a database of ~30
million non-coding RNA sequences. The 145'th nucleotide is masked. Can you guess
what it is?


I see this challenge as incredibly frustrating at times, but simultaneously,
this is the main reason why I find applying AI

Its Adenine (`A`) - the RNA BERT language model [1].


This type of work requires extensive use of analogy. For instance, we need to


Fundamental Gaps
1. left-to-right causal is a gap. There is no temporal or causal relationship
   between part of a biological sequence (protein or nucleic acid) that comes
  upstream

Ilya Sutskever says that next token prediction is an extremely powerful training
signal.

and claims that "the hardest problems in next token prediction is harder than
the hardest problem in bidirectional masked token prediction (i.e. BERT)"

This raises the question: is the same true of a modality like biological
sequences where


2. harder to inspect model: can't just generate sequences from the LM and
   "look" at them to see if the model is making sense. Or rather you can, but
turn-around times for such as task will take ~4-6 weeks (in teh good case)

3. Its impossible to inspect data to tell how good it is



4. Outlook

My oulook on the use of AI for biology is positive.





References
1. "An observation on Generalization" Ilya Sutskever (OpenAI)
2. "Borges and AI" Léon Bottou, Bernhardt Schölkopf (https://arxiv.org/abs/2310.01425)





