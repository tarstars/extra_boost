# TODO list for the extra boost project

## Problems and action items for article:

* unsteady flow of thoughts:
  * mark fragments of text with ideas they carry
  * devise global flow of ideas in the text
  * rewrite the text
  
* notions from the reviewers
  * why feature [t] can make things better? what is a theoretical foundations of our method? interaction between linear regression and the gbdt algorithm?
    * picture  illustrated that:
      * each leaf is a predicate
      * predicate is subset of dataset objects
      * subset leads to lift
      * lift is a function of time
    * put in the text:
      * since extra features contains constant, extra boost reproduces gbdt
      * first steps eliminate obvious dependencies, as gbdt does.
        Following steps take care of extra dependencies.
      
* extra neuronetwork
  * array of the linear regressors
  * multiply linear regressors with extra features
  * some neuronetwork on regular features
  * nonlinear interaction: use regular neuronetwork outputs as bias to extra features
    then put both them through some saturated nonlinearity, like tahn or sigmoid
    
* Extra Boost program TODO:
  * regression problem with MSE
  * refactoring:
    * Human-friendly interface
    * easy switching cpu-gpu
  * golang version of the algorithm
  * one-shot script for the reproducing results of the article
    * generate synthetic dataset
    * fit extra_boost regressor
    * apply regressor, draw the picture
    
* Non-synthetic dataset:
  * Finally decide, if we can do something interesting with HH dataset
    * Source of sine dependency in HH data set:
      * sort ids by presence in timeline
      * draw heatmap: vertical line - sorted ids, horizontal - time, brightness - presence of id in time
    * Is it possible to improve quality of HH dataset by adding data on presence of vacancy on site?
  * Any idea on constructing time-dependent dataset from the Yandex data?
  
  

