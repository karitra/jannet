#
# Alex Karev
#
# GNU v3 License
#
# Basic FF-BP network playground based on Krose "introduction..."
#
# TODO:
#
#   - implement brain damage strategy
#   - store/load network to disk
#   - correct weights initialization
#   - on-line, parallel training
#   - allocation tuning
#   - vectorized element wise operation
#
module Jannet

export sigm, dsigm, ftest
export FFBPNet
export sampleOnce!, sampleOnce, learnOnePattern

sigm(z; a=0.7) = 1 ./ (1 + exp(-a * z))
dsigm(y) = (1 - y) .* y
@inline ftest(x) = sin(x) / 2 + 0.5
sqrErr(y,p) = (y - p) .^2 / 2

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

	damage::Vector{T}

	function Layer(inp::Int, out::Int, momentum::Real)
		inp += 1

		@assert(inp > 1, "wrong numbers of input nodes")
		@assert(out >= 1, "wrong numbers of output nodes")

		#
		# Correct random initialization
		#
		# w  = Matrix{T}(out,inp)
		w  = MakeRandomLayer(T, inp, out)
		dw = MakeRandomLayer(T, inp, out) # Matrix{T}(out,inp)
		mw = MakeRandomLayer(T, inp, out) # Matrix{T}(out,inp)

		o = Vector{T}(out+1)
		o[1] = 1.0
		dmg = ones(T,out)

		err = Vector{T}(out)

		new( w, dw, mw, momentum, o, err, dmg)
	end
end

function setDamage!(l::Layer, idx::Int, v::Bool = true)
	l.damage[idx] = v ? 0.0 : 1.0
end

function clearDamage!(l::Layer)
	l.damage[:] = 1.0
end


typealias Layers{T} Vector{Layer{T}}

type FFBPNet{T<:Real}

	layers::Layers{T}
	act::Function

	inpCount::Int
	outCount::Int

	learningRate::T

	realType

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

		new(layers, act, inpCount, outCount, learningRate, T)
	end

end

function store(net::FFBPNet, fileName)

end

function load(net::FFBPNet, fileName)

end

#
# Set brain damage at layer's l (counting from input) output num
#
function setDamage!(net::FFBPNet, l::Int, k::Int, v::Bool = true)

	ll = l - 1 # real layer number, counting input as separate layer

	@assert(ll < length(net.layers) && ll >= 1 && length(net.layers) >= 3, "layer must be hidden one")
	@assert(k < length(net.layers[ll].out), "node id $k is incorrect")

	setDamage!(net.layers[ll], k, v)
end

function clearAllDamage!(net::FFBPNet, l::Int)
	
	ll = l - 1

	@assert(ll < length(net.layers) && ll >= 1 && length(net.layers) >= 3, "layer must be hidden one")
	clearDamage!(net.layers[l-1])
end

function sampleOnce!{T<:Real}(y::Vector{T}, net::FFBPNet{T}, x::Vector{T}; useBrainDamage::Bool = false)

	@assert( length(y) == net.outCount, "Wrong size of output vector, should be $(net.outCount)"    )	
	@assert( length(x) == net.inpCount + 1, "Wrong size of input vector, should be $(net.inpCount)" )

	a = x

	@inbounds for lr in net.layers

		ln = length(lr.out)

		# @show a
		# @show lr.out

		A_mul_B!(sub(lr.out, 2:ln), lr.W, a)
		broadcast!(net.act,lr.out,lr.out)

		#
		# apply 'brain damage' if any
		#
		if useBrainDamage && any(x -> x == 0, lr.damage)
			@fastmath @simd for k in eachindex(lr.damage)
				lr.out[k+1] *= lr.damage[k]
			end
		end

		a = lr.out
		a[1] = 1.0

		# @show a
	end

	y[:] = a[2:end]
end

function sampleOnce{T<:Real}(net::FFBPNet{T}, x::Vector{T}; useBrainDamage::Bool = false)
	y = Vector{T}(net.outCount)
	sampleOnce!(y, net, x, useBrainDamage=useBrainDamage)
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

function t3(t::Type;iters=100000, lr = 0.7, layout=[1 3 1], epsilon=2.3e-5, m=0.0, f=ftest)

	x = t[0:0.0005:1;]
	y = f(x * 2pi)

	nn = FFBPNet{t}(layout, learningRate = lr, momentum = m)

	idx = collect(1:length(x))

	shuffle!(idx)

	cvPart = floor(Int, length(idx) * 0.3)

	testIdx  = sub( idx, 1:cvPart )
	trainIdx = sub( idx, cvPart+1:length(idx) )

	@show length(testIdx)
	@show length(trainIdx)

	train_error = 0
	for k = 1:iters

		shuffle!(trainIdx)

		for i in trainIdx
			# @show x[i], y[i]
			learnOnePattern!( nn, t[1; x[i]], t[ y[i] ] )
		end

		tr_err = 0
		for i in trainIdx
			p = Jannet.sampleOnce(nn, t[1, x[i]])
   			tr_err .+= sqrErr(y[i],p)
   		end

        train_error = sum(tr_err) / length(trainIdx)

        println("tr_err($k) $train_error")

        if train_error < epsilon
        	println("break out earlier on $k iteration")
        	break
        end
	end

	@show train_error

	testError = 0
	@inbounds for i in testIdx
			p = Jannet.sampleOnce(nn, [1, x[i]])
   			testError .+= sqrErr(y[i],p)
	end

	testError = sum(testError) ./ length(testIdx)
	@show testError

	return nn
end

#
# Brain damage test
#
# Note: network should be trained
#
function t4(net::FFBPNet; f = ftest)

	n = length(net.layers) + 1
	if n < 3
		println("nets without hidden layer can't be damaged: no brain, no pain")
		return
	end

	#
	# Create test set
	# 
	x = net.realType[0:0.001:1;]
	y = f(x * 2pi)

	#
	# Construct hidden layers range
	#
	hiddenRange = 2:n-1

	minErrLayerId = -1
	minErrNodeId  = -1
	minCvError    = Inf

	for hl in hiddenRange
		for nodeId in eachindex(net.layers[hl-1].damage)

			setDamage!(net, hl, nodeId)

			idx = collect(1:length(x))
			shuffle!(idx)

			cvError = 0
			@fastmath @inbounds for i in idx
				p = sampleOnce(net, net.realType[1.0; x[i] ], useBrainDamage=true )
				cvError .+= sqrErr(p, y[i])
			end

			cvError = sum(cvError) ./ length(idx)

			if cvError < minCvError
				minErrNodeId  = nodeId
				minErrLayerId = hl
				minCvError    = cvError
			end

			setDamage!(net, hl, nodeId, false)
		end
	end


	minCvError, minErrLayerId, minErrNodeId
end


end
