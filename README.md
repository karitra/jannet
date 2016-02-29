# Jannet ANN's playground

 Basic (yet artificial, not ready for natural one) neural network implementation in Julia language. Written for studying purposes.

## Usage

```
julia> include("ffbp.jl")

julia> # in nn would be your brand new network
julia> nn = FFBPNet{Float32}( [3 10 2], learningRate=pi, momentum=0.2 );

```
First argument is a layout of network in form of 
> `[ <input size> <layer 1 size>... <layer K size> <output size> ]`,
where size corresponds to a number of nodes on the layer, layout organized from left to right, 
from input layer to the output. Type parameter of template could be any Real type.

### Train one sample
```
julia> x = Float32[ 1, 0.5, 0.1, 0 ]
julia> y = Float32[ 0, 1 ]
julia> learnOnePattern!( nn, x, y )
```
First vector is input with first bias activation element (should be equal to one), second is a desired output pattern vector

### Get the response
```
> p = Jannet.sampleOnce(nn, [1, x[i]])
```
In p would be the result for x pattern.  Jannet.sampleOnce! version exist.

## Tests

### Function approximation

Sample training for function f(x) = sin(x*2pi)/2 + 0.5 in interval for x:[0,1]  approximation:

```
julia> @time nn = Jannet.t3(Float64, iters=10000, lr=3, layout=[1 60 1], m = 0.3, epsilon=5e-6);
train_error = 9.063752343964018e-6
...
228.910095 seconds (377.72 M allocations: 14.130 GB, 3.72% gc time)
```
`iters` - count of iterations, can break out loop earlier on `train_error <= epsilon`, where `train_error` 
is average square error for training set.

Learning results of trained network can be visualized (checked) as follow:
```
julia> using Gadfly
...
julia> z = [ Jannet.sampleOnce(nn, Float32[1.0, x])[1] for x in 0:0.02:1 ];
julia> plot(y=z)
```
