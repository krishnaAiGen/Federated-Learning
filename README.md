# Weighted-Parameterization-FL

This research project modifies the widely used federated averaging algorithm using the concept of weighted average. The Federated averaging algorithm gives equal weights to the clients during aggregation if we are detecting cyber-attacks with FL. However, in real-world scenarios, malicious traffic is not uniformly distributed among all clients. Hence giving them equal weights makes convergence very slow. In this research project, we have developed a modified version of the fed-avg algorithm, which dynamically assigns distinct weights to each client during aggregation. These dynamic weights are determined with the help of malicious traffic present at each node. The higher the number of malicious traffic, the higher the weight of the client is. 

Our experimentation has shown that the re-parameterization of weights using the weighted-average concept makes convergence much faster than simple-average. 
<p style="color📘">This approach benefits the peer nodes in a distributed setting to learn the patterns of cyber-attacks happening on other peer nodes in a few communication rounds.
</p>

We tested our approach on CICIDS 2017 & NSL-KDD dataset. The blue graph represents the widely used federated averaging algorithm, and the green graph indicates our federated algorithm with reparameterization.


![Tux, the Linux mascot](/result/result.png)
