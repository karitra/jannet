#
# Alex Karev
#
# GNU v3 License
#
# Basic FF-BP network playground based on Krose B. "Introduction..."
#
# TODO:
#
#   - implement brain damage strategy (partly implemented)
#   - store/load network to disk
#   - correct weights initialization
#   - batch, parallel (multi-node) training
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
	convert( Matrix{t}, rand(-0.1:0.001:0.1,out, inp) )
end

type Layer{T<:Real}

	W::Matrix{T}
	dW::Matrix{T} #  gradient part of W
	# acc_dW::Matrix{T} # accumulated dW for batch processing

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

function applyAcc(ll::Layer)
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

	function FFBPNet(layout::Matrix{Int}; act=sigmoid, momentum = 0.0, learningRate = 0.4)

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

@inline function updateWeights!(layer::Layer)
	scale!(layer.mW, layer.momentum)

	@fastmath @inbounds @simd for k in eachindex(layer.W)
		layer.W[k] += layer.dW[k] + layer.mW[k]
		layer.mW[k] = layer.dW[k]
		# net.layers[i].acc_dW[k] += net.layers[i].dW[k]
	end
end

@inline function accWeights!(layer::Layer)
	scale!(layer.mW, layer.momentum)

	@fastmath @inbounds @simd for k in eachindex(layer.W)
		layer.acc_dW[k] += layer.dW[k] + layer.mW[k]
		layer.mW[k] = layer.dW[k]
		# net.layers[i].acc_dW[k] += net.layers[i].dW[k]
	end
end


function learnOnePattern!{T<:Real}(net::FFBPNet{}, x::Vector{T}, d::Vector{T}; batch = false)

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
		scale!(net.layers[i].dW, net.learningRate)
	end

	#
	# Updating weights
	#
	if batch
		@inbounds for i in eachindex(net.layers)
			accWeights!(net.layers[i])
		end
	else
		@inbounds for i in eachindex(net.layers)
			updateWeights!(net.layers[i])
		end
	end

end

