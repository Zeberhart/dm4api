Dialogue Management for API Search
==========

Dialogue Management for API Search (DM4API) is a project that models interactive API search in a standard dialogue system framework. The dialogue manager considers the strength of API search results and dialogue history to select an appropriate dialogue act response type. For more information, see the corresponding paper (under review).

This repository is organized into three folders:

* /dialogue_management
Code and data to implement the dialogue manager. Includes three policies (hand-crafted, learned, and baseline search), scripts to train and test policies, and data to construct API datasets for the Libssh and Allegro APIs. 

* /quantitative_eval
Materials related to the quantitative evaluation of the dialogue manager, including scripts and results.

* /human_study
Materials related to the quantitative evaluation of the dialogue manager, including the API search tool and results.
