# Semantic Segmentation of Extreme Climate Events

**Original Paper**: http://cs230.stanford.edu/projects_fall_2022/reports/40.pdf

**Original Codebase:** https://github.com/hannahg141/ClimateNet

**Authors**: Romain Lacombe, Hannah Grossman, Lucas Hendren, David Lüdeke

## What can I learn from this paper?

- [ ] Upsampling
- [ ] Semantic Segmentation
- [ ] [How to modify Loss functions for highly imbalanced models.](https://ieeexplore.ieee.org/document/9277638)

## Files

- `data/download_climatenet.ipynb` - this is almost entirely a fork of the original data extraction notebook. Nothing I wasn't too familiar with here so no need to look deeply into it.

## Paper Structure

It's been over a decade since I wrote my undergraduate thesis so might be useful to break down (at least for the first few papers I look at) the basic structure of the papers. I'll start very pedantically, but I imagine this will get more light-touch as similar themes crop up.

Also worth noting that I'm looking at the final projects for a Stanford **_module_**, and are therefore going to be much shorter than what I'll need to come up with.

### Abstract

- One sentence summarisation of context _("Climate action failure and extreme weather events are the two most severe global risks today.")_.
- One sentence overview of the methodology. Broken down into:.
  - What their model is maining to do: _("To advance automated detection of extreme weather events, ")_
  - The technique(s) they used: _("we have applied significant modifications to a novel light-weight context guided convolutional neural network, CGNet, and trained it for semantic segmentation of tropical cyclones and atmospheric rivers in climate data.")_
- Breakdown of the different things they tried _("feature engineering and augmentation, channel combinations, learning rate modifications, alternative loss functions, and architectural changes")_.
- What metrics they chose and why _("We specifically chose to focus on recall and sensitivity metrics, in contrast to previous approaches focusing on IoU (intersection over union), to penalize under-counting and optimize for identification of tropical cyclones")_.
- What worked: _("we found success in improving these metrics through the use of weighted loss functions,")_
- Future research / implications: _("We hope to contribute to improved automated extreme weather events detection models, which are of crucial importance for better attribution, prediction and mitigation of the impacts of climate change.")_.

### Introduction

- **First Paragraph:** High level overview of the problem, which cites **non-Machine Learning**, domain-relevant sources. Then goes on to describe the work so far which has been done using Deep Learning (& citing it). Paragraph finishes by citing the shortcomings of said research -- in this case reliance on heavy and complex architectures with
  huge numbers of parameters. This then neatly leads to the research question _("A key area of research is the development of lighter-weight architectures for semantic segmentation of tropical cyclones (TC) and atmospheric rivers (AR)")_.

- **Second paragraph** High level basis for this project:

  - The technique used (i.e. CNN).
  - What the input data is.
  - What the output data is.

- **Third Paragraph:**
  - List challenges specific to this task
  - List ways you tried to improve on the baseline (hyperparameters, augmentation, etc).
  - What worked?

### Related Work

Three examples of related work. The paragraphs follow a consistent structural pattern:

- **Reference to a Work or Study:** Each paragraph begins by referencing a specific work or study.
- **Description of the Work:** After introducing the work, go into specifics of the study, (methods or architecture used).
- **Outcome or Result:** Outcome or performance metrics of respective study.
- **Contextual Significance:** Paragraph concludes by highlighting importance or context of the work in relation to other studies or the current project.

### Dataset and Features

#### ClimateNet Dataset

- Where did you get the data from?
- How is it structured?
- How did you do the train / val / test split?
- How is the output structured? In this case its a map grid like the input, with class predictions on background weather (`0`) vs either of the extreme weather events.

#### Data Engineering

What features did you engineer? What was your rationale? How did you calculate them?

#### Data Augmentation

What did you try? What worked? What didn't work? Why!

### Methods

#### Baseline implementation

- What was the baseline, how did you train it.
- What training metric are you optimising for? Is there a contextual reason why you might want to optimise for more (e.g.) false negatives? In this case, false negatives very dangerous!
- How was the bias / variance / recall of your baseline?

#### CGNet Architecture

- Why did you go for the architecture you went for? Describe it, and its advantages.
- Were there any changes to the architecture you tried? Were they successful? Or were they not?

#### Loss Functions for Imbalanced Classes

- Explain the makeup of imbalanced classes.
- How are you accounting for this imbalance - **they used a review of loss functions which might come in handy**
  - [Shruti Jadon. A survey of loss functions for semantic segmentation (2020).](https://ieeexplore.ieee.org/document/9277638)

#### Metrics

- [ ] TODO: Work out why and how they used three different loss functions and success metrics -- isn't that too many?

* List which success metrics you used, and why.
* List which loss functions you used, and why.

### Experiments / Results / Discussion

Brief writeup of each of your experiments, and how they effected your success metrics. Are there any issues it threw up -- if so, how did you respond, and did that fix the issue?

How do did your tests affect the baseline?

Did you give more gravity to the results of certain class or metric? Why?

What problems did you run into? Were you able to fix any of them? Were some inherent limitations of your approach?

### Conclusion / Future Work

What can you draw from your experiments? What have you been able to prove? Even if you failed to reject the null hypothesis, doesn't that say something?

Finally, suggest at least two ways in which the experiement could be improved (but in a way which you couldn't just do yourself -- e.g. more/better data).
