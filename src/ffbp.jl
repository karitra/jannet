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
#   - try RPROP
#   - store/load network to disk
#
# TODO:
#
#   - check mini-batch algo as it converges extremely slow, suspect typo in implementation (or in tests)
#   - implement brain damage strategy (partly implemented)
#   - correct weights initialization
#   - mini-batch, parallel (multi-node) learning
#   - memory allocations tuning
#   - vectorized element wise operation
#
using JLD

export sigmoid, dsigm, ftest, sqrErr
export RPROPArgs, MakeNet, WeaveNetwork
export sampleOnce!, sampleOnce, learnOnePattern!, learnBatch!, setDamage!
export setAccDerivatives!

const ALPHA = 1.0

# @enum PrevSign WasMinus=-1 WasPlus=+1 WasChanges=0

sigmoid(z; a=ALPHA) = 1 ./ (1 + exp(-a * z))

function sigmVec!(y::Array, z::Array; alpha = ALPHA)

	@assert(length(y) == length(z), "sigmoid input and output must be of the same size")

	scale!(z,alpha)
	@fastmath @inbounds @simd for i in eachindex(z)
		y[i] = 1 / ( 1 + exp(z[i]) )
	end

end

dsigm(y) = ALPHA .* (1 - y) .* y
@inline ftest(x) = sin(x) / 2 + 0.5
sqrErr(y,p) = (y - p) .^2 / 2


function MakeRandomLayer(t::Type, inp::Int, out::Int)
	convert( Matrix{t}, rand(-0.02:1e-6:0.02, out, inp) )
	# convert( Matrix{t}, rand(-0.01:1e-3:0.01, out, inp) )
end

type Layer{T<:Real}

	W::Matrix{T}
	dW::Matrix{T} #  gradient part of W

	accD::Matrix{T} # accumulated dW for batch processing

	# RPROP part
	delta::Matrix{T} # current gradient step
	#prevSign::Matrix{Bool} # false : sign == -1, true sign == 1
	prevSign::Matrix{Float32} # TODO: check if it is slower then Float32 (weird things happens)

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
		delta = ones(T, out, inp) * 0.2
		sg  = zeros(Float32, out, inp)  #Matrix{Bool}(out, inp)

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

	RPROPArgs(use::Bool = false; minD = 1e-7, maxD = 60, etaMinus = 0.5, etaPlus = 1.2) = 
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

	epochsPassed::Int
end


function MakeNet(T::Type, layout::Matrix{Int}; act=sigmoid, momentum = 0.0, learningRate = 0.4, rprop = RPROPArgs() )

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

	FFBPNet{T}(layers, act, inpCount, outCount, learningRate, T, rprop, 0)
end

WeaveNetwork = MakeNet

function store(fileName::AbstractString, net::FFBPNet)
	jldopen(fileName, "w", compress=true) do out
		addrequire(out, Jannet)
		write(out, "layers", net.layers)
		write(out, "inputs", net.inpCount)
		write(out, "outputs", net.outCount)
		write(out, "learningRate", net.learningRate)
		write(out, "elType", net.realType)
		write(out, "rprop", net.rprop)
		write(out, "epochsPassed", net.epochsPassed)
	end
end

#
# Note: default constructed with sigmoid
#
function load(fileName::AbstractString)
	jldopen(fileName) do inp
		Jannet.FFBPNet(
			read(inp, "layers"),
			sigmoid,
			read(inp, "inputs"),
			read(inp, "outputs"),
			read(inp, "learningRate"),
			read(inp, "elType"),
			read(inp, "rprop"),
			read(inp, "epochsPassed") )
	end
end

function getClassErrors(nn::FFBPNet, X, Y)
 	err = 0
    errRate = 0
    rng = 1:size(Y,2)
    for i in rng
          y = Jannet.sampleOnce(nn, X[:,i] )
          err += sum((y - Y[:,i]) .^2 / 2)
          _, sampleId = findmax(y)
          _, patternId = findmax(Y[:,i])
           if sampleId != patternId
                errRate += 1
           end
     end
    err /= length(rng)
    errRate /= length(rng)
    err,errRate,1-errRate
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
	@assert( length(x) == net.inpCount, "Wrong size of input vector, should be $(net.inpCount)" )

	a = x

	@inbounds for (i,lr) in enumerate(net.layers)

		ln = length(lr.out)

		# @show a
		# @show lr.out

		if i == 1
			#
			# Add weights of the bias separately in case of the first layer in order to get reed of bias in input vector
			#
			rows, cols = size(lr.W)

			w = sub(lr.W, :, 2:cols)
			A_mul_B!(sub(lr.out, 2:ln), w, a)

			# add bias (first column)
			@fastmath @simd for k in 2:ln 
				lr.out[k] += lr.W[k-1,1] 
			end
		else
			A_mul_B!(sub(lr.out, 2:ln), lr.W, a)
		end
		
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

@inline function setAccDerivatives!(lr::Layer, acc::Matrix)
	@fastmath @inbounds @simd for k in eachindex(lr.accD)
		lr.accD[k] = acc[k]
	end
end

function setAccDerivatives!(net::FFBPNet, accs::Array)
	for k in eachindex(accs)
		setAccDerivatives!(net.layers[k], accs[k])
	end
end


function applyRPROPDelta!(layer::Layer, rprop::RPROPArgs)

	@fastmath @inbounds @simd for k in eachindex(layer.W)

		if layer.accD[k] != 0.0
			sn = sign(layer.accD[k])

			signState = sn * layer.prevSign[k]

			if signState > 0 # sign the same
				layer.delta[k] = min(rprop.etaPlus  * layer.delta[k], rprop.maxDelta)
				layer.W[k] += sn * layer.delta[k]
				layer.prevSign[k] = sn
			elseif signState < 0  # sign changed
				layer.delta[k] = max(rprop.etaMinus * layer.delta[k], rprop.minDelta)
				layer.prevSign[k] = 0
			else  # sign changed in previous step
				layer.W[k] += sn * layer.delta[k]
				layer.prevSign[k] = sn
			end

			# layer.W[k] += sn * layer.delta[k]
			layer.accD[k] = 0
		end

	end
end

@inline function applyRPROPDelta!(net::FFBPNet)
	@inbounds for i in eachindex(net.layers)
		applyRPROPDelta!(net.layers[i], net.rprop)
	end
end

function applyBatchDelta!(layer::Layer, learningRate::Real)

	@assert(count > 0, "number of samples must be greater then zero")

	scale!(layer.mW, layer.momentum)

	@fastmath @inbounds @simd for k in eachindex(layer.W)
		layer.W[k] += learningRate * layer.accD[k] + layer.mW[k]
		layer.mW[k] = layer.accD[k]
		layer.accD[k] = 0
	end
end


@inline function applyBatchDelta!(net::FFBPNet)
	for i in eachindex(net.layers)
		applyBatchDelta!(net.layers[i], net.learningRate, count = count)
	end
end

@inline function applyBatchDelta!(netDst::FFBPNet, netSrc::FFBPNet)
	for i in eachindex(netDst.layers)
		applyBatchDelta!(netDst.layers[i], netSrc.layers[i].acc_dW)
	end
end


#
# Samples in matrix is in column order, so iterate column-wise (seems to be a bit faster (not proved) and require less allocation)
#
function learnBatch!{R<:Real}(net::FFBPNet, X::Matrix{R}, Y::Matrix{R}; m::Int = -1, applyReduceStep = true)
	if m == -1
		m = size(X,2)
	end

	@assert( m <= size(X,2) && m <= size(Y,2), "in input (X) and output (Y) should be the same numbers of samples (m, columns)" )

	# @show m

	@inbounds for i in 1:m
		learnOnePattern!(net, X[:,i], Y[:,i], batch = true)
	end

	net.epochsPassed += 1

	if applyReduceStep
		if net.rprop.useRPROP
			applyRPROPDelta!(net)
		else
			applyBatchDelta!(net)
		end
	else
		return map( lr -> lr.accD, net.layers )
	end
end

function learnOnePattern!{R<:Real}(net::FFBPNet{R}, x::AbstractArray{R}, d::AbstractArray{R}; batch = false)

	y = sampleOnce(net, x)

	lln = length(net.layers)

	@assert(lln > 0, "Wrong number of layers, network is inconsistent")

	for i in lln:-1:1

		# @shouldw i

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
		# @show net.layers[i].out

		if i == 1
			yi = x

			rows, cols = size(net.layers[i].dW)

			dw = sub(net.layers[i].dW, :, 2:cols)

			# @show i, size(yi), rows, cols
			A_mul_Bt!(dw, delta, yi)

			@inbounds @simd for k in 1:rows
				net.layers[i].dW[k,1] = delta[k]
			end
		else
			yi = net.layers[i-1].out
			A_mul_Bt!(net.layers[i].dW, delta, yi)
		end
		
		if !batch
			# if batch, 'll scale later
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

