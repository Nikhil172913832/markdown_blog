#ml 

1. Used for data that is sequential
2. The key differentiator between them and other nn are they take time and sequence into account, they have a temporal(that is kind of time)  dimension.
3. Can also be used for image as images can be broken down into a series of patches(i.e making parts) and those patches can be used as a sequence.
4. Feed forward nets pass information only once through each node whereas in recurrent nets cycle it through a loop.
5. Feed forward network has no notion of time, and the only input it considers is the current example it has been exposed to. 
6. RNN are similar in notion to a Markov process where decision at time step t-1 affects the decision at time step t.
7. So it is kinda like they have a sort of memory as there recent past decision affects their current decision and the reason to have it is for when the sequence of information also matters like text or audio , video or even a photo as mentioned in point 3.
8. So as each decision is affected by the previous one so it has a cascading effect which helps finding correlation between different events and these correlations are called "long-term dependencies" as event downstream in time depends upon, and is a function of, one or more events that came before. So, we can think of them like they share weights over time.
9. So to get a better analogy we can think of recurrent nets like human memory and its affects can be thought of like how our past experiences affect our present same way weights at time steps before t will have an effect on the weights at time step t.
10. So to define it mathematically we use :
	![[Pasted image 20240710102558.png]]
11. Here **h<sub>t</sub>** refers to the hidden state at time step t **x<sub>t</sub>** is the input at time step t , **W** is the weight and **U** is the transition matrix and the weight matrix is like a filer which determines how much importance has to be given to the past hidden state and the present inputs and then the function φ is either a logistic sigmoid or tanh  
12. So as these feedback loops occur at each time step so they contain certain traces of previous time step until the memory persists so it not only depends on the previous time step but also the one that precede the previous time step.
### Back Propagation through time
1. So time in this case is the ordered series of calculations performed so basically an ordered series of time steps.
2. Now see it is not practical to do BPTT for longer sequences as then you will have to keep gradient and activation of a lot of time steps in memory to be able to perform the operation so it would make much more sense to break the steps into smaller chunks and then do the forward and backward passes as it would be computationally much easier and feasible for the least and we call this as Truncated BPTT which simply means to break the longer sequences into smaller chunks and then doing BPTT which also help in frequently updating each parameter which helps in faster convergence but there is a cache as we are dividing the sequence we are limiting the scope of long term dependencies.
3. Now comes the problem of vanishing gradients which means the gradients become too small as we are back propagating which makes it difficult to make meaningful updates.

## LSTM
1. They basically help preserve the error from vanishing gradients by keeping a constant error and allow recurrent nets to learn over many time steps.
2. So these have a sort of a memory which they keep in gated cell unlike the hidden state of recurrent nets which are modified every time so they control the basic crud of that memory and they are analog as it is differentiable unlike digital which makes it suitable for back propagation.
3. So 