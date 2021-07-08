# oopiest-neural-network-ever
I would appreciate any suggestions on how to test and improve this code that I wrote in vanila python

hello!

tell me how to make it faster and more accurate and tell me my mistakes I would be realy grateful
contact me on "sadrat83@gmail.com"
thank you

two explanations on this program:
1_ in network initialization the "all_layers" _that you pass into it_ is a list of numbers like [2,5,5] or [1,6,6,6] which refers to layers. to translate it, [2,5,5] means our network has a layer which contains two neurons and a layer which contains five neurons and another layer wich contains five neurons. Beware of that this layers DO NOT CONTAIN THE INPUT LAYER! because input layer differs in initialization.
2_ The next parameter that you pass into when initializing the network is " n_inputs " which tells our network how many neurons are in our input layer(how many inputs do we have). for e.g if you want to feed forward [0.9, 0.53, 0.2] you should pass 3 as number of inputs.
