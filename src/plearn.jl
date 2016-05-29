#
# author Karev A.
# description: Part of Jannet library. Parallel (multi-node) learning in batch mode (RPROP)
#
using DistributedArrays

summAcc(lr1, lr2) = [ lr1[i] + lr2[i] for i in eachindex(lr1) ]

function parLearnBatch!{T<:Real}(net::Jannet.FFBPNet{T}, iters::Int, X::Matrix{T}, Y::Matrix{T}, m::Int = -1)

	if m == -1
		m = size(X,2)
	end

	@assert(net.rpop.useRPROP, "only RPROP batch mode supported")
	@assert( m <= size(X,2) && m <= size(Y,2), "in input (X) and output (Y) should be the same numbers of samples (m, columns)" )

	n = size(X,1)
	println("Init of distributed X array...")
	Xdist = DArray( (n,m), workers(), [1, length(workers()) ]) do I
		r1,r2 = I
		x = zeros(net.realType, length(r1), length(r2))
		x[:,:] = X[r1,r2]
	end

	cl = size(Y,1)
	println("Init of distributed Y array...")
	Ydist = DArray( (cl,m), workers(), [1, length(workers()) ]) do I
		r1,r2 = I
		y = zeros(net.realType, length(r1), length(r2))
		y[:,:] = Y[r1, r2]
	end

	for i in 1:iters

		tic()

		acc = @parallel (+) for i in workers()

			Xlocal = localpart(Xdist)
			Ylocal = localpart(Ydist)

			Jannet.learnBatch!(net, Xlocal, Ylocal, applyReduceStep = false)
		end

		Jannet.setAccDerivatives!(net, acc)
		Jannet.applyRPROPDelta!(net)

		elapsed = toq()

	    train_err = 0
	    n,m = size(X)
        @fastmath @inbounds for k in 1:m
           ysample = Jannet.sampleOnce(net, X[:,k])
           train_err += sum(Jannet.sqrErr(ysample, Y[:,k]))
        end

        @assert(m > 0, "Incorrect number of samples")

        train_err /= m
   
   		@printf("parEpoch(%d) %.3f tr_er %.6f\n", i, elapsed, train_err)
	end

	# close(Xdist::DArray)
	# close(Ydist::DArray)
end
