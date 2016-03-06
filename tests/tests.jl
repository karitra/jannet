module Tests

export t3,t4

using Jannet

################################################################################
#
# Tests section
#
################################################################################
function t1()
	nn = Jannet.FFBPNet{Float32}([1 2 1], learningRate=0.3)
	
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

	nn = Jannet.FFBPNet{Float64}([1 2 1], learningRate=0.1)

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

#
# Note: after testing with few 'brain damage' iteration appropriate network for test function approximation is 
#
# julia>  @time nn = Jannet.t3(Float32, iters=1000000, lr=5, layout=[1 5 6 1], m = 0.05, epsilon=1e-5);
# ...
# tr_err(694) 0.00017526299
# tr_err(695) 3.531719e-5
# tr_err(696) 4.446271e-5
# tr_err(697) 4.1777654e-5
# tr_err(698) 1.5803449e-5
# tr_err(699) 1.9088016e-5
# tr_err(700) 1.2926726e-5
# tr_err(701) 9.97425e-6
# break out earlier on 701 iteration
# train_error = 9.97425f-6
# testError = 1.1541664f-5
#  21.486778 seconds (60.96 M allocations: 2.387 GB, 7.31% gc time)
#
function t3(t::Type;iters=1000000, lr = 0.5, layout=[1 5 6 1], epsilon=1e-6, m=0.05, f=ftest)

	x = t[0:0.0005:1;]
	y = f(x * 2pi)

	nn = Jannet.FFBPNet{t}(layout, learningRate = lr, momentum = m)

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

		tic()

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
        elapsed = toq()

        @printf("iter(%d) %.6f sec\t tr_err %.8f\n", k, elapsed, train_error)

        if train_error < epsilon
        	println("break out earlier on $k iteration")
        	break
        end
	end

	@show train_error

	test_error = 0
	@inbounds for i in testIdx
			p = Jannet.sampleOnce(nn, [1, x[i]])
   			test_error .+= sqrErr(y[i],p)
	end

	test_error = sum(test_error) ./ length(testIdx)
	@show test_error

	return nn
end

#
# Brain damage test
#
# Note: network should be trained
#
function t4(net::Jannet.FFBPNet; f = ftest)

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

function t5(t::Type; iters=100, lr = 0.5, layout=[1 5 6 1], epsilon=1e-6, m=0.05, f=ftest, rprop=Jannet.RPROPArgs())

	x = t[0:0.0005:1;]
	y = f(x * 2pi)

	idx = collect(1:length(x))

	nn = Jannet.FFBPNet{t}(layout, learningRate = lr, momentum = m, rprop = rprop)

	# @show length(testIdx)
	# @show length(trainIdx)

	for k = 1:iters

		shuffle!(idx)

		testPart = floor(Int, length(idx) * 0.3)

		testIdx  = sub( idx, 1:testPart )
		trainIdx = sub( idx, testPart+1:length(idx) )

		bias = ones(t,length(trainIdx),1);

		X = [ bias x[trainIdx] ]'
		Y = y[trainIdx]'

		# @show size(X)
		# @show size(Y)

		tic()

		Jannet.learnBatch!(nn, X, Y, size(X,2) )

		elapsed = toq()

		test_error = 0
		@inbounds for i in testIdx
				p = Jannet.sampleOnce(nn, [1, x[i] ])
	   			test_error .+= sqrErr( y[i], p)
		end

		test_error = sum(test_error) ./ length(testIdx)

		@printf("batch(%d) elapsed %.6f sec. \ttest_error %.8f\n", k, elapsed, test_error)

		if test_error < epsilon
			println("break out earlier on $k iteration")
			break;
		end
	end

	return nn
end

end