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
julia> @time nn = Jannet.t3(Float32, iters=1000000, lr=4, layout=[1 50 50 1], m = 0.03, epsilon=1e-5);
...
break out earlier on 4860 iteration
train_error = 9.92018f-6
testError = 8.857888f-6
522.754328 seconds (436.08 M allocations: 16.246 GB, 1.19% gc time)
```
`iters` - count of iterations, can break out loop earlier on `train_error <= epsilon`, where `train_error` 
is average square error for training set.

Learning results of trained network can be visualized (checked) as follow:
```
julia> using Gadfly
...
julia> y = [ Jannet.sampleOnce(nn, Float32[1.0, x])[1] for x in 0:0.02:1 ];
julia> ysample= Jannet.ftest(0:0.124:1* 2pi)
...
julia> draw( PNG("sample.png", 22cm,12cm), plot( layer(y=ysample, Geom.line), layer(y=y, Geom.point, Theme(default_color=colorant"green")), layer(y=(y-ysample).^2*100, Geom.bar, Theme(default_color=colorant"dark red") ) ) )
```
Square error rate for sample is shown in red color bars (scaled by 100), sample results are in green dots, and blue line as function itself: ![sample plot](sample.png)