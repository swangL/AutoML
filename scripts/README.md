|Shorthand|Experiment desc.| Findings|
|:--|:--|:--|
|c|Vanilla controller| prone to go towards termination since this the most enforced part (when REINFORCE the first action gets all enforcement)
|cc|Collected Controller: have a pool of **all** actions not just the block way of building|
|g| Gru enabled instead of LSTM|
|not| No termination during start only valid for *c*|
|antit| Anti termination scheme to force deeper networks|
|ent| Entropy enabled|
|REG| this is a **dataset** trying regression method by utilizing R^2 value!!!|
