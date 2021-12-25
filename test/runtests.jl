include("../src/WJFUNet.jl")
WJFUNet.trainMatchingUNet("../VideoDatasets", 5, 2, 16, 10, 10)
