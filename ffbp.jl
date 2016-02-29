#
# Alex Karev
#
# Basic FF-BP network playground based on Krose "introduction..."
#
# TODO:
#
#   - store/load network to disk
#   - correct weights initialization
#   - on-line, parallel training
#   - allocation tuning
#   - vectorized element wise operation
#
module Jannet

using StatsBase

export sigm, dsigm, FFBPNet, sampleOnce!, sampleOnce

sigm(z; a=1.0) = 1 ./ (1 + exp(-a * z))
dsigm(y) = (1 - y) .* y
ftest(x) = sin(x) / 2 + 0.5

function MakeRandomLayer(t::Type, inp::Int, out::Int)
	convert( Matrix{t}, rand(-0.1:0.001:0.1,out, inp) )
end

type Layer{T<:Real}

	W::Matrix{T}
	dW::Matrix{T} # sigma W

	mW::Matrix{T} # momentum W (aka dW(t-1))
	momentum::T

	out::Vector{T}
	err::Vector{T}

	function Layer(inp::Int, out::Int, momentum::Real)
		inp += 1
		
		#
		# Correct random initialization
		#
		# w  = Matrix{T}(out,inp)
		w  = MakeRandomLayer(T, inp, out)
		dw = MakeRandomLayer(T, inp, out) # Matrix{T}(out,inp)
		mw = MakeRandomLayer(T, inp, out) # Matrix{T}(out,inp)

		o = Vector{T}(out+1)
		o[1] = 1.0

		err = Vector{T}(out)

		new( w, dw, mw, momentum, o, err)
	end

end

typealias Layers{T} Vector{Layer{T}}

type FFBPNet{T<:Real}

	layers::Layers{T}
	act::Function

	inpCount::Int
	outCount::Int

	learningRate::T

	function FFBPNet(layout::Matrix{Int}; act=sigm, momentum = 0.0, learningRate = 0.4)
		
		if (length(layout) < 2)
			error("layout must containg at least one layer: [in <hidden..> out]")
		end

		layers = Layers{T}()
		inpCount = layout[1]
		outCount = layout[end]

		for k in 2:length(layout)
			nl = Layer{T}( layout[k-1], layout[k], momentum )
			push!(layers, nl )
		end

		new(layers, act, inpCount, outCount, learningRate)
	end

end

function sampleOnce!{T<:Real}(y::Vector{T}, net::FFBPNet{T}, x::Vector{T})

	@assert( length(y) == net.outCount, "Wrong size of output vector, should be $(net.outCount)"    )	
	@assert( length(x) == net.inpCount + 1, "Wrong size of input vector, should be $(net.inpCount)" )

	a = x

	for lr in net.layers

		ln = length(lr.out)

		# @show a
		# @show lr.out

		A_mul_B!(sub(lr.out, 2:ln), lr.W, a)
		broadcast!(net.act,lr.out,lr.out)
		
		a = lr.out
		@inbounds a[1] = 1.0

		# @show a
	end

	y[:] = a[2:end]
end

function sampleOnce{T<:Real}(net::FFBPNet{T}, x::Vector{T})
	y = Vector{T}(net.outCount)
	sampleOnce!(y, net, x)
end

function learnOnePattern!{T<:Real}(net::FFBPNet{}, x::Vector{T}, d::Vector{T})

	y = sampleOnce(net, x)

	ll = length(net.layers)

	@assert(ll > 0, "Wrong number of layers, network is inconsistent")

	for i in ll:-1:1

		# @show i

		if i == ll # spacial treat to output layer
			
			#
			# TODO: remove temporary allocations, if any
			# original code: 
			# net.layers[i].err = (d - y) .* dsigm(y)
			
			@fastmath @inbounds @simd for k in eachindex(y)
				net.layers[i].err[k] = (d[k] - y[k]) * dsigm(y[k])
			end
 		else
			n,m = size(net.layers[i+1].W)
			w = sub(net.layers[i+1].W, :, 2:m)

			At_mul_B!(net.layers[i].err, w, net.layers[i+1].err)

			# TODO: remove temporary allocations, if any
			yln = length(net.layers[i].out)
			y   = sub( net.layers[i].out, 2:yln)

			# original code:
			# net.layers[i].err .*= dsigm(y)
			@fastmath @inbounds @simd for k in eachindex(net.layers[i].err)
				net.layers[i].err[k] *= dsigm( y[k] )
			end
		end

		delta = net.layers[i].err

		# @show delta

		yi = i == 1 ? x : net.layers[i-1].out

		A_mul_Bt!(net.layers[i].dW, delta, yi)
		scale!(net.layers[i].dW, net.learningRate)
	end

	#
	# Updating weights
	#
	for i in eachindex(net.layers)

		scale!(net.layers[i].mW, net.layers[i].momentum)

		@fastmath @inbounds @simd for k in eachindex(net.layers[i].W)
			net.layers[i].W[k] += net.layers[i].dW[k] + net.layers[i].mW[k]
			net.layers[i].mW[k] = net.layers[i].dW[k]
		end

		#@show net.layers[i].W
	end

end

################################################################################
#
# Tests section
#
################################################################################

function t1()
	nn = FFBPNet{Float32}([1 2 1], learningRate=0.3)
	
	nn.layers[1].W[1,1] = 0.5 
	nn.layers[1].W[2,1] = 0.5
	nn.layers[1].W[1,2] = 0.1
	nn.layers[1].W[2,2] = 0.7

	nn.layers[2].W[1,1] = 0.5
	nn.layers[2].W[1,2] = 3
	nn.layers[2].W[1,3] = 7

	y = sampleOnce(nn, Float32[1.0; pi])

	@show nn.layers[1].W
	@show nn.layers[1].out
	@show nn.layers[2].W
	@show nn.layers[2].out

	@show y
end

function t2(iters=1)

	x = Float64[0:0.001:pi;]
	y = ftest(x)

	nn = FFBPNet{Float64}([1 2 1], learningRate=0.1)

	for k in 1:iters
		for i in eachindex(x)
			
			# @show x[i]
			# @show y[i]

			# @show nn.layers[1].W
			# @show nn.layers[2].W

			learnOnePattern!( nn, Float64[1.0; x[i]], Float64[ y[i] ] )

			# @show nn.layers[1].W
			# @show nn.layers[1].err
			# @show nn.layers[2].W
			# @show nn.layers[2].err
		end
	end

	# y = sampleOnce(nn, [ 1, x[1] ])
	# @show y, sin(x[1])
	# y = sampleOnce(nn, [ 1, x[2] ])
	# @show y, sin(x[2])
	# y = sampleOnce(nn, [ 1, x[3] ])
	# @show y, sin(x[3])

	# y = sampleOnce(nn, [ 1, x[end] ])
	# @show y, sin(x[end])
	# k = div( length(x), 2) + 1
	# y = sampleOnce(nn, [ 1, x[k] ] )
	# @show y, sin(x[k])
end

function t3(t::Type;iters=100000, lr = 0.7, layout= [1 3 1], epsilon=2.3e-5, m = 0.0)

	x = t[0:0.001:1;]
	y = ftest(x * 2pi)

	nn = FFBPNet{t}(layout, learningRate=lr, momentum=m)

	train_error = 0
	for k = 1:iters

		idx = sample(1:length(x), floor(Int,length(x) * 0.7) )
		for i in idx
			# @show x[i], y[i]
			learnOnePattern!( nn, t[1; x[i]], t[ y[i] ] )
		end

		tr_err = 0
		for i in idx
			p = Jannet.sampleOnce(nn, [1, x[i]])
   			tr_err .+= (y[i] - p) .^2 / 2
   		end

        train_error = sum(tr_err) / length(idx)

        # @show train_error

        if train_error < epsilon
        	println("break out earlier on $k iteration")
        	break
        end
	end

	@show train_error

	y1 = sampleOnce(nn, t[1; 0])
	y2 = sampleOnce(nn, t[1; 0.2]) 
	y3 = sampleOnce(nn, t[1; 0.25])
	y4 = sampleOnce(nn, t[1; 0.5]) 
	y5 = sampleOnce(nn, t[1; 0.75])
	y6 = sampleOnce(nn, t[1; 1])

	@show y1,y2
	@show y3
	@show y4
	@show y5
	@show y6

	return nn
end

end
