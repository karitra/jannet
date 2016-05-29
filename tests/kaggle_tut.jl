
using Images
# using DataFrames

using Jannet
using JLD

const IMAGE_SIZE = 400
const LABELS_SIZE = 62

const AVG_ERR_STEP = 20

@inline mkMapping() = [ Char(c) => i for (i,c) in enumerate([ '0':'9'; 'a':'z'; 'A':'Z' ]) ]
@inline mkRevMapping() = [ c for c in [ '0':'9'; 'a':'z'; 'A':'Z' ] ]

@inline showNorm(net::Jannet.FFBPNet) = for lr in net.layers @show norm(lr.W) end

function loadAllImages(path, d::Dict; imgSize = IMAGE_SIZE)

	imgs = Vector(0)

	for fname in readdir(path)
		m = match(r"^(\d+)\.\w+\.sample\.Bmp$", fname)

		pth = path * "/" * fname

		# continue
		
		if m == nothing || !isfile(pth) 
			# println("no match $fname")
			continue
		end		

		img = float32( load(pth))
		imgMat = mean(separate(img),3)

		id = parse(Int, m.captures[1] )
		push!(imgs, ( d[id], reshape(imgMat, 1, imgSize), id ) )
	end

	@printf("read %d images from %s\n", length(imgs), path)

	imgs
end

function loadLabelsMapping(file)

	d = Vector(0)

	open(file) do inp
		for ln in drop( eachline(inp), 1)
			tk = split(ln, [',', ' '])
			id = parse(Int, first(tk) )
			push!( d, ( id, chomp(last(tk)) ) )
		end
	end

	d
end

function LoadLearnImages(paths::Vector, mapFiles::Vector; imgSize::Int = IMAGE_SIZE)
	
	md = Vector(0)

	for fl = mapFiles
		m = loadLabelsMapping(fl)
		append!(md,m)
	end

	@show length(md)

	d = [ id => class for (id,class) in md ]

	imgs = Vector(0)

	for p in paths
		imSet = loadAllImages(p,d, imgSize = imgSize)
		append!(imgs, imSet)
	end

	imgs
end

type DataSet{T<:Real}
	X::Matrix{T}
	Y::Matrix{T}
	cl::Vector{Char}
	ids::Vector{Int}
end

function storeDataSet(fileName::AbstractString, ds::DataSet)
	jldopen(fileName, "w", compress=true) do out
		write(out, "DataSet", ds)
	end
end

function loadDataSet(fileName::AbstractString)
	jldopen(fileName, "r") do inp
		return read(inp, "DataSet")
	end

	# DataSet(X,Y,cl,ids);
end

function MakeSamples(t::Type, imgs, n, m, r)
	X = zeros(t,n,m)
	Y = zeros(t,n,r)

	cl  = Vector{Char}(n)
	ids = Vector{Int}(n)
	
	clMap = mkMapping()

	for (i,img) in enumerate(imgs)
           X[i,:] = img[2].data[:]

           c = first(img[1])
           Y[i, clMap[ c ] ] = 1

           cl[i] = c
           ids[i] = img[3]
    end

	DataSet{t}(X',Y',cl,ids)
end

function LoadKaggleSamples(t::Type, paths::Vector, mapFiles::Vector; imgSize::Int = IMAGE_SIZE) 
	imgs = LoadLearnImages(paths, mapFiles, imgSize = imgSize)
	MakeSamples(t, imgs, length(imgs), imgSize, LABELS_SIZE)
end

LearnKaggleSet(net::Jannet.FFBPNet, iters::Int, ds::DataSet; batchMode = true) =
	LearnKaggleSet(net, iters, ds.X, ds.Y, batchMode = batchMode)

function LearnKaggleSet(net::Jannet.FFBPNet, iters::Int, X, Y; batchMode = true)
	
	idx = collect(1:size(X,2))
    testSize = floor(Int, 0.3 * length(idx))
    
    prev_error = 0
    min_error = Inf
    min_error_it = 0
    
    avg_error = 0
    acc_error = 0 

	trainRange = testSize+1:length(idx)

	Xsample = zeros(eltype(X), size(X,1), length(trainRange))
    Ysample = zeros(eltype(X), size(Y,1), length(trainRange))

    for i in 1:iters

    	shuffle!(idx)
    	
		testIdx  = idx[1:testSize]
    	trainIdx = idx[trainRange]
    
    	count   = fill(30, size(Y,1)) # patterns counts of each type to feed to network
    	fillIdx = 0

    	# @show length(trainIdx)

    	for j in trainIdx

			fillIdx += 1

    		Xsample[:,fillIdx] = X[:,j]
    		Ysample[:,fillIdx] = Y[:,j]

    		# mx = findfirst(Y[:,j])

    		# @show j, idx[j], mx, count[mx]

    		# if count[mx] > 0 
    		# 	fillIdx += 1

    		# 	Xsample[:,fillIdx] = X[:,j]
    		# 	Ysample[:,fillIdx] = Y[:,j]

    		# 	count[mx] -= 1
    		# end
    	end

    	# @show fillIdx

    	if batchMode
	    	tic()
	        # Jannet.learnBatch!(net, Xsample, Ysample, fillIdx)
	        Jannet.learnBatch!(net, X, Y)
	        elapsed = toq()
	    else
	    	tic()
    		for m in 1:fillIdx
    			Jannet.learnOnePattern!(net, Xsample[:,m], Ysample[:,m], batch = batchMode)
    		end
    		net.epochsPassed += 1
    		elapsed = toq()
    	end

        test_err = 0
        err_rate = 0
        @fastmath @inbounds for k in testIdx
           ysample = Jannet.sampleOnce(net, X[:,k])
           test_err += sum(Jannet.sqrErr(ysample, Y[:,k]))
           err_rate += last(findmax(Y[:,k])) != last(findmax(ysample))
        end

        test_err /= length(testIdx)
        err_rate /= length(testIdx)

        delta_err = prev_error - test_err
        prev_error = test_err
        acc_error += test_err

        if test_err < min_error
        	min_error = test_err
        	min_error_it = i
        end

        if i % AVG_ERR_STEP == 0
        	avg_error = acc_error / AVG_ERR_STEP
        	acc_error = 0
        end

        @printf("epoch(%d) %.3fs \ttest_err %.7f  min_err(%d) %.6f avg_tr_err(%d) %.6f d_err(prev) %.6f\n",
        	i, elapsed, test_err, min_error_it, min_error, AVG_ERR_STEP, avg_error, delta_err)
    end

    println("\tnet learned $(net.epochsPassed) epoch(s)")
end

function LearnKaggleSetGroup(net::Jannet.FFBPNet, iters::Int, ds::DataSet; batchMode = true)
	Y = ds.Y
	X = ds.X
	Ygrp = zeros(net.realType, 3, size(Y,2))

	alphLen = length('a':'z')

	println("Preparing groups mapping")

	@inbounds for k in 1:size(Y,2)
		_,i = findmax(Y[:,k])
		if i <= 10
			Ygrp[1,k] = 1
		elseif i <= alphLen + 10
			Ygrp[2,k] = 1
		else
			Ygrp[3,k] = 1
		end
	end

	LearnKaggleSet(net, iters, X, Ygrp, batchMode = batchMode)
end

function PrepareTrainChunks(X,Y,M,parts)
	
	Xsample = Vector(0)
	Ysample = Vector(0)
	offset = 1

	for i in 1:parts
		Xchunk = X[:,offset:parts:M]
		Ychunk = Y[:,offset:parts:M]
		push!(Xsample, Xchunk)
		push!(Ysample, Ychunk)
		offset += 1
	end

	Xsample, Ysample
end

function LearnByChunks(nn, Xset, Yset, iters=1, innerloop=1)
    for i in 1:iters
        for j in eachindex(Xset)
            LearnKaggleSet(nn, innerloop, Xset[j], Yset[j], batchMode=false)
        end
    end
end

function WriteResult(nn::Jannet.FFBPNet, csvFile::AbstractString, Xtest, idsTest, revMap)
	open(csvFile, "w") do out
		write(out, "ID,Class\n")
		for i in eachindex(idsTest)
		    y = sampleOnce(nn, Xtest[:,i]); _,mxId = findmax(y)
		    write(out, "$(idsTest[i]),$(revMap[mxId])\n")
		end
	end
end

