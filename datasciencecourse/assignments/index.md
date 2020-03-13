---
layout: page
title: Assignments
---

All assignments for the class will be listed here.

- Five homework assignments, each with 2-3 programming problems.
- A midterm "tutorial" assignment where you will write up a short tutorial on a data science subject.
- A final project, done in groups, on a data science problem of your choosing.

All assignments will be released by 11:59 PM ET on the release date, and are due at 11:59pm ET (midnight) on the due date.

You are expected to know and adhere to the [course policies](/policies), which govern late days, submissions, and collaboration.

## Assignment dates

We may occasionally modify assignment dates and scopes. If we do that, there will be an announcement in-class and an update here.

| | Release date | Due date |
| --- | --- | --- |
| [Homework 1](homework-1/)      | Aug 26 | Sep 12 |
| [Homework 2](homework-2/)      | Sep 13 | Oct 1  |
| [Homework 3](homework-3/)      | Oct 2  | Oct 24 |
| [Tutorial Proposal](tutorial/) | Sep 16 | Sep 27 |
| [Tutorial](tutorial/)          |        | Oct 15 |
| [Tutorial Evaluation](tutorial/)| Oct 24| Oct 29 |
| [Homework 4](homework-4/)      | Oct 25 | Nov 11 |
| [Homework 5](homework-5/)      | Nov 12 | Nov 25 |
| [Final Project Proposal](project/) | Oct 21 | Nov 1  |
| [Final Project Video](project/)    |        | Dec 5  |
| [Final Project Feedback](project/) | Dec 6  | Dec 6  |
| [Final Project Report](project/)   |        | Dec 11 |

TAs may not be available to answer questions about an assignment after its due date; keep this in mind before deciding to use your grace days.

## Homework

Homeworks are distributed as Jupyter notebooks, submitted for auto-grading via [Diderot](http://www.diderot.one/).

To get access to the course materials, go to [Diderot](http://www.diderot.one/) and register using your `andrew.cmu.edu` account. Select our course "15-388 Practical Data Science" and use the code `35603` to register.

## Writing Code

The five homeworks in this course are all auto-graded programming assignments. Here is some information to get you started:

### Environment

Begin by setting up the environment: you need Python 3.6.7 or later, PIP (Python package manager) for that version, and Git (version control software) installed.

Environments are notoriously difficult to debug, especially if grown slowly over time. While we encourage you to experiment with setting your environment up, we will only provide technical support for these three configurations:

**Vagrant** If you want a ready-to-use environment, take a look at [our custom Vagrantfile](https://github.com/gauravmm/datascienceenv). Vagrant is a popular tool that allows you to rapidly provision (set up) a virtual machine and we have written a configuration file (and tutorial) that prepares everything for the course. This is the easiest method.

**Anaconda** Download and install it [here](https://www.anaconda.com/distribution/).

**Windows Subsystem for Linux** If you are running Windows, consider using the WSL. Begin by following [these instructions](https://www.maketecheasier.com/install-linux-subsystem-for-windows10/) to install WSL. Once you have created your account, run this to install the prerequisite system packages:

```shell
sudo apt update
sudo apt upgrade -yq
sudo apt install -yq python3 python3-pip git
```

Your linux home directory is located at either `\\wsl$\home\<linux-username>` (WSL2) or `C:\Users\%USERNAME%\AppData\Local\Packages\CanonicalGroupLimited.UbuntuonWindows_79rhkp1fndgsc\LocalState\rootfs\home\<linux-username>` (WSL1). Create a shortcut to this folder.

Move the homework handouts to this folder to make them accessible to the Ubuntu app. You can upload the completed `.ipynb` files directly from this folder.

### Extraction

Extract each `.tgz` file using any archive extractor. Each archive contains:

- `requirements.txt`, which details the Python dependencies;
- `[...].ipynb`, which has the questions in an IPython Notebook; and
- additional data files which are needed to run the notebook and should not be submitted.

Begin by installing the dependencies in `requirements.txt`. With pip, you can use `pip3 -r requirements.txt`. Then you can start the Jupyter notebook server with the command:

- `jupyter notebook <path-to-folder>`, or
- `python3 -m jupyter notebook <path-to-folder>`

For more help on running a Jupyter notebook, you can view the Jupyter [quick start guide](http://jupyter-notebook-beginner-guide.readthedocs.io/en/latest/execute.html).

Our question files contain tests to allow you to rapidly check that your own code is working. We score your submissions based on a more extensive set of tests.

## Submitting

You can submit each homework by uploading the `.ipynb` file through the Diderot website. Do not rename the file or alter any function signatures; we rely on these for automatic grading.

You should get a score breakdown after a few minutes.

## Tutorial

In lieu of a midterm exam, students will write a tutorial on a data science topic of their choosing.  More information will be posted here when the assignment is released.  Again, **no late days are permitted** on the tutorial, and failure to submit by the deadline will result in zero points.

## Final project

The final project of the course will consist of a large data science project done in teams of 2-3 people (single person or four person teams will be considered on an individual basis).  The final report for this project will be a Jupyter notebook detailing the data collection, analysis, and results.  In addition to the report, teams will also prepare a short video for showing during a final project video session.  More information and dates is available on the [project page](project/).

**No late days are permitted** on the final project, and failure to submit by the deadline will result in zero points.
