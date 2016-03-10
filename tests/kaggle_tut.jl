
using Images
# using DataFrames

using Jannet

const IMAGE_SIZE = 400
const LABELS_SIZE = 62

const AVG_ERR_STEP = 20

@inline mkMapping() = [ Char(c) => i for (i,c) in enumerate([ '0':'9'; 'a':'z'; 'A':'Z' ]) ]
@inline mkRevMapping() = [ c for c in [ '0':'9'; 'a':'z'; 'A':'Z' ] ]

@inline showNorm(net::FFBPNet) = for lr in net.layers @show norm(lr.W) end

function loadAllImages(path, d::Dict)

	imgs = Vector(0)

	for fname in readdir(path)
		m = match(r"^(\d+)\.sample\.Bmp$", fname)

		pth = path * "/" * fname

		# continue
		
		if m == nothing || !isfile(pth) 
			# println("no match $fname")
			continue
		end		

		img = float32( load(pth))
		imgMat = mean(separate(img),3)

		id = parse(Int, m.captures[1] )
		push!(imgs, ( d[id], reshape(imgMat, 1, IMAGE_SIZE), id ) )
	end

	@printf("read %d images from %s", length(imgs), path)

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

function LoadLearnImages(paths::Vector, mapFiles::Vector)
	
	md = Vector(0)

	for fl = mapFiles
		m = loadLabelsMapping(fl)
		append!(md,m)
	end

	@show length(md)

	d = [ id => class for (id,class) in md ]

	imgs = Vector(0)

	for p in paths
		imSet = loadAllImages(p,d)
		append!(imgs, imSet)
	end

	imgs
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

	X',Y',cl,ids
end

function LoadKaggleSamples(t::Type, paths::Vector, mapFiles::Vector) 
	imgs = LoadLearnImages(paths, mapFiles)
	MakeSamples(t, imgs, length(imgs), IMAGE_SIZE, LABELS_SIZE)
end


function LearnKaggleSet(net::Jannet.FFBPNet, iters::Int, X, Y)
	
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

    	for j in trainIdx

    		mx = findfirst(Y[:,j])

    		# @show j, idx[j], mx, count[mx]

    		if count[mx] > 0 
    			fillIdx += 1

    			Xsample[:,fillIdx] = X[:,j]
    			Ysample[:,fillIdx] = Y[:,j]

    			count[mx] -= 1
    		end
    	end

    	# @show fillIdx

    	tic()
        Jannet.learnBatch!(net, Xsample, Ysample, fillIdx)
        elapsed = toq()

        test_err = 0
        @fastmath @inbounds for k in testIdx
           ysample = Jannet.sampleOnce(net, X[:,k])
           test_err += sum(Jannet.sqrErr(ysample, Y[:,k]))
        end

        test_err /= length(testIdx)
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

        @printf("epoch(%d) %.4f s \ttest_err %.7f  min_err(%d) %.6f avg_tr_err(%d) %.6f d_err(prev) %.6f\n",
        	i, elapsed, test_err, min_error_it, min_error, AVG_ERR_STEP, avg_error, delta_err)
    end

end
