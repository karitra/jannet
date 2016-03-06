#
# Alex Karev
#
# GNU v3 License
#
# Basic FF-BP network playground based on Krose B. "Introduction..."
#
# DONE (aka feature):
#
#   - on-line learning
#   - mini-batch propagation (only subset of patterns in a time processed)
#
# TODO:
#
#   - check mini-batch algo as it converges extremely slow, suspect typo in implementation (or in tests)
#   - implement brain damage strategy (partly implemented)
#   - store/load network to disk
#   - correct weights initialization
#   - mini-batch, parallel (multi-node) learning
#   - try RPROP
#   - memory allocations tuning
#   - vectorized element wise operation
#
export sigmoid, dsigm, ftest, sqrErr
export FFBPNet
export sampleOnce!, sampleOnce, learnOnePattern!, setDamage!

sigmoid(z; a=0.7) = 1 ./ (1 + exp(-a * z))

function sigmVec!(y::Array, z::Array; alpha = 0.7)

	@assert(length(y) == length(z), "sigmoid input and output must be of the same size")

	scale!(z,alpha)
	@fastmath @inbounds @simd for i in eachindex(z)
		y[i] = 1 / ( 1 + exp(z[i]) )
	end

end

dsigm(y) = (1 - y) .* y
@inline ftest(x) = sin(x) / 2 + 0.5
sqrErr(y,p) = (y - p) .^2 / 2

function MakeRandomLayer(t::Type, inp::Int, out::Int)
	convert( Matrix{t}, rand(-0.1:0.001:0.1, out, inp) )
end

type Layer{T<:Real}

	W::Matrix{T}
	dW::Matrix{T} #  gradient part of W

	accD::Matrix{T} # accumulated dW for batch processing

	# RPROP part
	delta::Matrix{T} # current gradient step
	prevSign::Matrix{Bool} # false : sign == -1, true sign == 1

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
		dw = zeros(T, out, inp)
		accD = zeros(T, out, inp)

		#
		# PROP part, may be not used
		#
		delta = ones(T, out, inp)
		sg  = Matrix{Bool}(out, inp)

		mw = zeros(T, out, inp)

		o = Vector{T}(out+1)
		o[1] = 1.0
		dmg = ones(T,out)

		err = Vector{T}(out)

		new( w, dw, accD, delta, sg, mw, momentum, o, err, dmg)
	end
end

function applyAcc(ll::Layer)
end

function setDamage!(l::Layer, idx::Int, v::Bool = true)
	l.damage[idx] = v ? 0.0 : 1.0
end

function clearDamage!(l::Layer)
	l.damage[:] = 1.0
end


typealias Layers{T} Vector{Layer{T}}

type RPROPArgs

	useRPROP::Bool

	minDelta::Real
	maxDelta::Real

	etaMinus::Real
	etaPlus::Real

	RPROPArgs(use::Bool = false; minD = 1e-9, maxD = 50, etaMinus = 0.6, etaPlus = 1.5) = 
		new(use, minD, maxD, etaMinus, etaPlus)
end

type FFBPNet{T<:Real}

	layers::Layers{T}
	act::Function

	inpCount::Int
	outCount::Int

	learningRate::T

	realType

	rprop::RPROPArgs

	function FFBPNet(layout::Matrix{Int}; act=sigmoid, momentum = 0.0, learningRate = 0.4, rprop = RPROPArgs() )

		if (length(layout) < 2)
			error("layout must contain at least one layer: [in <hidden..> out]")
		end

		layers = Layers{T}()
		inpCount = layout[1]
		outCount = layout[end]

		for k in 2:length(layout)
			nl = Layer{T}( layout[k-1], layout[k], momentum )
			push!(layers, nl )
		end

		new(layers, act, inpCount, outCount, learningRate, T, rprop)
	end

end

function store(net::FFBPNet, fileName)

end

function load(net::FFBPNet, fileName)

end

#
# Set brain damage at layer's lr (counting from input) output num
#
function setDamage!(net::FFBPNet, lr::Int, k::Int, v::Bool = true)

	ll = lr - 1 # real layer number, counting input as separate layer

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

@inline function updateWeights!(layer::Layer)
	scale!(layer.mW, layer.momentum)

	@fastmath @inbounds @simd for k in eachindex(layer.W)
		layer.W[k] += layer.dW[k] + layer.mW[k]
		layer.mW[k] = layer.dW[k]
	end
end

@inline function accDerivative!(layer::Layer)
	@fastmath @inbounds @simd for k in eachindex(layer.W)
		layer.accD[k] += layer.dW[k]
	end
end


function applyRPROPDelta!(layer::Layer, rprop::RPROPArgs)

	@fastmath @inbounds @simd for k in eachindex(layer.W)

		sn = sign(layer.accD[k])

		layer.delta[k] = ( (sn < -0 && layer.prevSign[k] == false) || (sn > +0 && layer.prevSign[k] == true) ) ?
			min(rprop.etaPlus  * layer.delta[k], rprop.maxDelta) :
			max(rprop.etaMinus * layer.delta[k], rprop.minDelta)

		# save sign of gradient for next iteration
		layer.prevSign[k] = sn > 0 ? true : false
		
		layer.W[k] += sn * layer.delta[k]
		layer.accD[k] = 0
	end
end

@inline function applyRPROPDelta!(net::FFBPNet)
	@inbounds for i in eachindex(net.layers)
		applyRPROPDelta!(net.layers[i], net.rprop)
	end
end

function applyBatchDelta!(layer::Layer, learningRate::Real ;count::Int = 1)

	@assert(count > 0, "number of samples must be greater then zero")

	scale!(layer.mW, layer.momentum)

	@fastmath @inbounds @simd for k in eachindex(layer.W)

		layer.W[k] += (learningRate * layer.accD[k] + layer.mW[k]) / count
		layer.mW[k] = layer.accD[k]

		layer.dW[k]   = 0
		layer.accD[k] = 0
	end
end

@inline function applyBatchDelta!(layer::Layer, accD::Matrix)
	@assert(size(layer.dW) == size(accD), "accumulated gradient must be of the same size as weights matrix")

	@fastmath @inbounds @simd for k in eachindex(layer.W)
		layer.W[k] += accD[k]
	end
end

@inline function applyBatchDelta!(net::FFBPNet; count::Int = 1)
	for i in eachindex(net.layers)
		applyBatchDelta!(net.layers[i], net.learningRate, count = count)
	end
end

@inline function applyBatchDelta!(netDst::FFBPNet, netSrc::FFBPNet)
	for i in eachindex(netDst.layers)
		applyBatchDelta!(netDst.layers[i], netSrc.layers[i].acc_dW)
	end
end


function parLearnBatch!{T<:Real}(net::FFBPNet{T}, X::Matrix{T}, Y::Matrix{T}, m::Int = -1)
	if m == -1
		m = size(X,2)
	end

	@assert( m <= size(X,2) && m <= size(Y,2), "in input (X) and output (Y) should be the same numbers of samples (m, columns)" )




end

#
# Samples in matrix is in column order, so iterate column-wise (seems to be a bit faster (not proved) and require less allocation)
#
function learnBatch!{T<:Real}(net::FFBPNet{T}, X::Matrix{T}, Y::Matrix{T}, m::Int = -1)

	if m == -1
		m = size(X,2)
	end

	@assert( m <= size(X,2) && m <= size(Y,2), "in input (X) and output (Y) should be the same numbers of samples (m, columns)" )

	@inbounds for i in 1:m
		learnOnePattern!(net, X[:,i], Y[:,i], batch = true)
	end

	if net.rprop.useRPROP
		applyRPROPDelta!(net)
	else
		applyBatchDelta!(net)
	end
end


function learnOnePattern!{T<:Real}(net::FFBPNet{T}, x::Vector{T}, d::Vector{T}; batch = false)

	y = sampleOnce(net, x)

	lln = length(net.layers)

	@assert(lln > 0, "Wrong number of layers, network is inconsistent")

	for i in lln:-1:1

		# @show i

		if i == lln # spacial treat to output layer
			
			#
			# TODO: remove temporary allocations, if any
			# original code: 
			# net.layers[i].err = (d - y) .* dsigm(y)
			# Note: broadcast seems a little bit slower then SIMD marked loop, but reduce code noise
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
			# Note: broadcast seems a little bit slower then SIMD marked loop, but reduce code noise
			@fastmath @inbounds @simd for k in eachindex(net.layers[i].err)
				net.layers[i].err[k] *= dsigm( y[k] )
			end
		end

		delta = net.layers[i].err

		# @show delta

		yi = i == 1 ? x : net.layers[i-1].out

		A_mul_Bt!(net.layers[i].dW, delta, yi)

		if !batch
			# if batch 'll scale later
			scale!(net.layers[i].dW, net.learningRate)
		end
	end

	#
	# Updating weights
	#
	if batch
		@inbounds for i in eachindex(net.layers)
			accDerivative!(net.layers[i])
		end
	else
		@inbounds for i in eachindex(net.layers)
			updateWeights!(net.layers[i])
		end
	end

end

