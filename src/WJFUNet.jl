module WJFUNet
using PyCall
using ImageIO
using VideoIO
using ColorTypes
using ImageShow
using ImageTransformations
using Random
using WJFParallelTask

export distortImage

function backup()
    frames = VideoIO.load("../VideoDatasets/F01.mp4")
    numFrames = length(frames)
    (numRows, numCols) = size(frames[1])

    numSamples = 20
    samples = [
            RGB{Float32}.(frames[rand(1:numFrames)]) for idxSample = 1:numSamples
        ]

    tf = pyimport("tensorflow")
    keras = tf.keras
    models = keras.models
    layers = keras.layers
    optimizers = keras.optimizers

    inputs = layers.Input
    #struct WjfUNet
    #end
end

function computePixel(
    ltp::RGB{Float32}, rtp::RGB{Float32},
    lbp::RGB{Float32}, rbp::RGB{Float32},
    wRow::Float32, wCol::Float32,
    )::RGB{Float32}
    ltw = (1.0 - wRow) * (1.0 - wCol)
    rtw = (1.0 - wRow) * wCol
    lbw = wRow * (1.0 - wCol)
    rbw = wRow * wCol
    r = ltw * ltp.r + rtw * rtp.r + lbw * lbp.r + rbw * rbp.r
    g = ltw * ltp.g + rtw * rtp.g + lbw * lbp.g + rbw * rbp.g
    b = ltw * ltp.b + rtw * rtp.b + lbw * lbp.b + rbw * rbp.b
    return RGB{Float32}(r,g,b)
end

function computePixel(
    imageInput::Matrix{RGB{Float32}},
    row::Float32,
    col::Float32,
    )::RGB{Float32}
    (numRows, numCols) = size(imageInput)
    rowPos = min(max(row, 1.0f0), numRows)
    colPos = min(max(col, 1.0f0), numCols)
    rowInt = Int(floor(rowPos))
    colInt = Int(floor(colPos))
    if rowInt == numRows
        if colInt == numCols
            return imageInput[rowInt, colInt]
        else
            wCol = Float32(colPos - colInt)
            wRow = 1.0f0
            return computePixel(
                imageInput[rowInt - 1, colInt],
                imageInput[rowInt - 1, colInt + 1],
                imageInput[rowInt, colInt],
                imageInput[rowInt, colInt + 1],
                wRow, wCol
            )
        end
    else
        wRow = Float32(rowPos - rowInt)
        if colInt == numCols
            wCol = 1.0f0
            return computePixel(
                imageInput[rowInt, colInt - 1],
                imageInput[rowInt, colInt],
                imageInput[rowInt + 1, colInt - 1],
                imageInput[rowInt + 1, colInt],
                wRow, wCol
            )
        else
            wCol = Float32(colPos - colInt)
            return computePixel(
                imageInput[rowInt, colInt],
                imageInput[rowInt, colInt + 1],
                imageInput[rowInt + 1, colInt],
                imageInput[rowInt + 1, colInt + 1],
                wRow, wCol
            )
        end
    end
end

function computeTriangleArea(
    row0::Float32,
    col0::Float32,
    row1::Float32,
    col1::Float32,
    row2::Float32,
    col2::Float32,
    )::Float32
    return 0.5f0 * abs((row1 - row0) * (col2 - col0) -
        (row2 - row0) * (col1 - col0))
end

function computeFirstRowUpTriangle!(
    imageOutput::Matrix{RGB{Float32}},
    movements::Matrix{Tuple{Float32,Float32}},
    colBegin::Int,
    colEnd::Int,
    sourceCellCenter::Tuple{Float32,Float32},
    destCellCenter::Tuple{Float32,Float32},
    imageInput::Matrix{RGB{Float32}},
    )
    for row = 1:Int(destCellCenter[1])
        w = (row - 1) / (destCellCenter[1] - 1)
        colBeginOfRow = Int(ceil((1.0 - w) * colBegin + w * destCellCenter[2]))
        colEndOfRow = Int(floor((1.0 - w) * colEnd + w * destCellCenter[2]))

        for col = colBeginOfRow:colEndOfRow
            wUp = computeTriangleArea(
                1.0f0, Float32(colBegin),
                1.0f0, Float32(colEnd),
                Float32(row), Float32(col),
            )
            wLeft = computeTriangleArea(
                1.0f0, Float32(colBegin),
                Float32(row), Float32(col),
                destCellCenter[1], destCellCenter[2],
            )
            wRight = computeTriangleArea(
                1.0f0, Float32(colEnd),
                Float32(row), Float32(col),
                destCellCenter[1], destCellCenter[2],
            )
            wAll = wUp + wLeft + wRight
            rowSource = (wUp * sourceCellCenter[1] +
                wLeft * 1.0f0 + wRight * 1.0f0) / wAll
            colSource = (wUp * sourceCellCenter[2] +
                wLeft * Float32(colEnd) + wRight * Float32(colBegin)) / wAll
            imageOutput[row, col] = computePixel(
                imageInput, rowSource, colSource,
            )
            movements[row, col] = (
                rowSource - Float32(row), colSource - Float32(col),
            )
        end
    end
end

function computeFirstRowDownTriangle!(
    imageOutput::Matrix{RGB{Float32}},
    movements::Matrix{Tuple{Float32,Float32}},
    colTop::Int,
    sourceCellCenter1::Tuple{Float32,Float32},
    destCellCenter1::Tuple{Float32,Float32},
    sourceCellCenter2::Tuple{Float32,Float32},
    destCellCenter2::Tuple{Float32,Float32},
    imageInput::Matrix{RGB{Float32}},
    )
    @assert destCellCenter1[1] == destCellCenter2[1]
    for row = 1:Int(destCellCenter1[1])
        w = (row - 1) / (destCellCenter1[1] - 1)
        colBeginOfRow = Int(ceil((1.0 - w) * colTop + w * destCellCenter1[2]))
        colEndOfRow = Int(floor((1.0 - w) * colTop + w * destCellCenter2[2]))
        for col = colBeginOfRow:colEndOfRow
            wLeft = computeTriangleArea(
                1.0f0, Float32(colTop),
                Float32(row), Float32(col),
                destCellCenter1[1], destCellCenter1[2],
            )
            wRight = computeTriangleArea(
                1.0f0, Float32(colTop),
                Float32(row), Float32(col),
                destCellCenter2[1], destCellCenter2[2],
            )
            wDown = computeTriangleArea(
                Float32(row), Float32(col),
                destCellCenter1[1], destCellCenter1[2],
                destCellCenter2[1], destCellCenter2[2],
            )
            wAll = wLeft + wRight + wDown
            rowSource = (wLeft * sourceCellCenter2[1] +
                wRight * sourceCellCenter1[1] + wDown * 1.0f0) / wAll
            colSource = (wLeft * sourceCellCenter2[2] +
                wRight * sourceCellCenter1[2] + wDown * colTop) / wAll
            imageOutput[row, col] = computePixel(
                imageInput, rowSource, colSource,
            )
            movements[row, col] = (
                rowSource - Float32(row), colSource - Float32(col),
            )
        end
    end
end

function computeLastRowUpTriangle!(
    imageOutput::Matrix{RGB{Float32}},
    movements::Matrix{Tuple{Float32,Float32}},
    colBottom::Int,
    sourceCellCenter1::Tuple{Float32,Float32},
    destCellCenter1::Tuple{Float32,Float32},
    sourceCellCenter2::Tuple{Float32,Float32},
    destCellCenter2::Tuple{Float32,Float32},
    imageInput::Matrix{RGB{Float32}},
    )
    (numRows, numCols) = size(imageOutput)
    @assert destCellCenter1[1] == destCellCenter2[1]
    for row = Int(destCellCenter1[1]):numRows
        w = (Float32(numRows) - row) / (Float32(numRows) - destCellCenter1[1])
        colBeginOfRow = Int(ceil((1.0 - w) * colBottom + w * destCellCenter1[2]))
        colEndOfRow = Int(floor((1.0 - w) * colBottom + w * destCellCenter2[2]))
        for col = colBeginOfRow:colEndOfRow
            wLeft = computeTriangleArea(
                Float32(numRows), Float32(colBottom),
                Float32(row), Float32(col),
                destCellCenter1[1], destCellCenter1[2],
            )
            wRight = computeTriangleArea(
                Float32(numRows), Float32(colBottom),
                Float32(row), Float32(col),
                destCellCenter2[1], destCellCenter2[2],
            )
            wUp = computeTriangleArea(
                Float32(row), Float32(col),
                destCellCenter1[1], destCellCenter1[2],
                destCellCenter2[1], destCellCenter2[2],
            )
            wAll = wLeft + wRight + wUp
            rowSource = (wLeft * sourceCellCenter2[1] +
                wRight * sourceCellCenter1[1] + wUp * Float32(numRows)) / wAll
            colSource = (wLeft * sourceCellCenter2[2] +
                wRight * sourceCellCenter1[2] + wUp * colBottom) / wAll
            imageOutput[row, col] = computePixel(
                imageInput, rowSource, colSource,
            )
            movements[row, col] = (
                rowSource - Float32(row), colSource - Float32(col),
            )
        end
    end
end

function computeLastRowDownTriangle!(
    imageOutput::Matrix{RGB{Float32}},
    movements::Matrix{Tuple{Float32,Float32}},
    colBegin::Int,
    colEnd::Int,
    sourceCellCenter::Tuple{Float32,Float32},
    destCellCenter::Tuple{Float32,Float32},
    imageInput::Matrix{RGB{Float32}},
    )
    (numRows, numCols) = size(imageOutput)
    for row = Int(destCellCenter[1]):numRows
        w = (Float32(numRows) - row) / (Float32(numRows) - destCellCenter[1])
        colBeginOfRow = Int(ceil((1.0 - w) * colBegin + w * destCellCenter[2]))
        colEndOfRow = Int(floor((1.0 - w) * colEnd + w * destCellCenter[2]))

        for col = colBeginOfRow:colEndOfRow
            wDown = computeTriangleArea(
                Float32(numRows), Float32(colBegin),
                Float32(numRows), Float32(colEnd),
                Float32(row), Float32(col),
            )
            wLeft = computeTriangleArea(
                Float32(numRows), Float32(colBegin),
                Float32(row), Float32(col),
                destCellCenter[1], destCellCenter[2],
            )
            wRight = computeTriangleArea(
                Float32(numRows), Float32(colEnd),
                Float32(row), Float32(col),
                destCellCenter[1], destCellCenter[2],
            )
            wAll = wDown + wLeft + wRight
            rowSource = (wDown * sourceCellCenter[1] +
                wLeft * Float32(numRows) + wRight * Float32(numRows)) / wAll
            colSource = (wDown * sourceCellCenter[2] +
                wLeft * Float32(colEnd) + wRight * Float32(colBegin)) / wAll
            imageOutput[row, col] = computePixel(
                imageInput, rowSource, colSource,
            )
            movements[row, col] = (
                rowSource - Float32(row), colSource - Float32(col),
            )
        end
    end
end

function computeFirstColLeftTriangle!(
    imageOutput::Matrix{RGB{Float32}},
    movements::Matrix{Tuple{Float32,Float32}},
    rowBegin::Int,
    rowEnd::Int,
    sourceCellCenter::Tuple{Float32,Float32},
    destCellCenter::Tuple{Float32,Float32},
    imageInput::Matrix{RGB{Float32}},
    )
    for col = 1:Int(destCellCenter[2])
        w = (col - 1) / (destCellCenter[2] - 1)
        rowBeginOfCol = Int(ceil((1.0 - w) * rowBegin + w * destCellCenter[1]))
        rowEndOfCol = Int(floor((1.0 - w) * rowEnd + w * destCellCenter[1]))

        for row = rowBeginOfCol:rowEndOfCol
            wLeft = computeTriangleArea(
                Float32(rowBegin), 1.0f0,
                Float32(rowEnd), 1.0f0,
                Float32(row), Float32(col),
            )
            wUp = computeTriangleArea(
                Float32(rowBegin), 1.0f0,
                Float32(row), Float32(col),
                destCellCenter[1], destCellCenter[2],
            )
            wDown = computeTriangleArea(
                Float32(rowEnd), 1.0f0,
                Float32(row), Float32(col),
                destCellCenter[1], destCellCenter[2],
            )
            wAll = wLeft + wUp + wDown
            rowSource = (wLeft * sourceCellCenter[1] +
                wUp * Float32(rowEnd) + wDown * Float32(rowBegin)) / wAll
            colSource = (wLeft * sourceCellCenter[2] +
                wUp * 1.0f0 + wDown * 1.0f0) / wAll
            imageOutput[row, col] = computePixel(
                imageInput, rowSource, colSource,
            )
            movements[row, col] = (
                rowSource - Float32(row), colSource - Float32(col),
            )
        end
    end
end

function computeFirstColRightTriangle!(
    imageOutput::Matrix{RGB{Float32}},
    movements::Matrix{Tuple{Float32,Float32}},
    rowLeft::Int,
    sourceCellCenter1::Tuple{Float32,Float32},
    destCellCenter1::Tuple{Float32,Float32},
    sourceCellCenter2::Tuple{Float32,Float32},
    destCellCenter2::Tuple{Float32,Float32},
    imageInput::Matrix{RGB{Float32}},
    )
    @assert destCellCenter1[2] == destCellCenter2[2]
    for col = 1:Int(destCellCenter1[2])
        w = (col - 1) / (destCellCenter1[2] - 1)
        rowBeginOfCol = Int(ceil((1.0 - w) * rowLeft + w * destCellCenter1[1]))
        rowEndOfCol = Int(floor((1.0 - w) * rowLeft + w * destCellCenter2[1]))
        for row = rowBeginOfCol:rowEndOfCol
            wUp = computeTriangleArea(
                Float32(rowLeft), 1.0f0,
                Float32(row), Float32(col),
                destCellCenter1[1], destCellCenter1[2],
            )
            wDown = computeTriangleArea(
                Float32(rowLeft), 1.0f0,
                Float32(row), Float32(col),
                destCellCenter2[1], destCellCenter2[2],
            )
            wRight = computeTriangleArea(
                Float32(row), Float32(col),
                destCellCenter1[1], destCellCenter1[2],
                destCellCenter2[1], destCellCenter2[2],
            )
            wAll = wUp + wDown + wRight
            rowSource = (wUp * sourceCellCenter2[1] +
                wDown * sourceCellCenter1[1] + wRight * rowLeft) / wAll
            colSource = (wUp * sourceCellCenter2[2] +
                wDown * sourceCellCenter1[2] + wRight * 1.0f0) / wAll
            imageOutput[row, col] = computePixel(
                imageInput, rowSource, colSource,
            )
            movements[row, col] = (
                rowSource - Float32(row), colSource - Float32(col),
            )
        end
    end
end

function computeLastColLeftTriangle!(
    imageOutput::Matrix{RGB{Float32}},
    movements::Matrix{Tuple{Float32,Float32}},
    rowRight::Int,
    sourceCellCenter1::Tuple{Float32,Float32},
    destCellCenter1::Tuple{Float32,Float32},
    sourceCellCenter2::Tuple{Float32,Float32},
    destCellCenter2::Tuple{Float32,Float32},
    imageInput::Matrix{RGB{Float32}},
    )
    (numRows, numCols) = size(imageOutput)
    @assert destCellCenter1[2] == destCellCenter2[2]
    for col = Int(destCellCenter1[2]):numCols
        w = (Float32(numCols) - col) / (Float32(numCols) - destCellCenter1[2])
        rowBeginOfCol = Int(ceil((1.0 - w) * rowRight + w * destCellCenter1[1]))
        rowEndOfCol = Int(floor((1.0 - w) * rowRight + w * destCellCenter2[1]))
        for row = rowBeginOfCol:rowEndOfCol
            wUp = computeTriangleArea(
                Float32(rowRight), Float32(numCols),
                Float32(row), Float32(col),
                destCellCenter1[1], destCellCenter1[2],
            )
            wDown = computeTriangleArea(
                Float32(rowRight), Float32(numCols),
                Float32(row), Float32(col),
                destCellCenter2[1], destCellCenter2[2],
            )
            wLeft = computeTriangleArea(
                Float32(row), Float32(col),
                destCellCenter1[1], destCellCenter1[2],
                destCellCenter2[1], destCellCenter2[2],
            )
            wAll = wUp + wDown + wLeft
            rowSource = (wUp * sourceCellCenter2[1] +
                wDown * sourceCellCenter1[1] + wLeft * rowRight) / wAll
            colSource = (wUp * sourceCellCenter2[2] +
                wDown * sourceCellCenter1[2] + wLeft * Float32(numCols)) / wAll
            imageOutput[row, col] = computePixel(
                imageInput, rowSource, colSource,
            )
            movements[row, col] = (
                rowSource - Float32(row), colSource - Float32(col),
            )
        end
    end
end

function computeLastColRightTriangle!(
    imageOutput::Matrix{RGB{Float32}},
    movements::Matrix{Tuple{Float32,Float32}},
    rowBegin::Int,
    rowEnd::Int,
    sourceCellCenter::Tuple{Float32,Float32},
    destCellCenter::Tuple{Float32,Float32},
    imageInput::Matrix{RGB{Float32}},
    )
    (numRows, numCols) = size(imageOutput)
    for col = Int(destCellCenter[2]):numCols
        w = (Float32(numCols) - col) / (Float32(numCols) - destCellCenter[2])
        rowBeginOfCol = Int(ceil((1.0 - w) * rowBegin + w * destCellCenter[1]))
        rowEndOfCol = Int(floor((1.0 - w) * rowEnd + w * destCellCenter[1]))

        for row = rowBeginOfCol:rowEndOfCol
            wRight = computeTriangleArea(
                Float32(rowBegin), Float32(numCols),
                Float32(rowEnd), Float32(numCols),
                Float32(row), Float32(col),
            )
            wUp = computeTriangleArea(
                Float32(rowBegin), Float32(numCols),
                Float32(row), Float32(col),
                destCellCenter[1], destCellCenter[2],
            )
            wDown = computeTriangleArea(
                Float32(rowEnd), Float32(numCols),
                Float32(row), Float32(col),
                destCellCenter[1], destCellCenter[2],
            )
            wAll = wUp + wDown + wRight
            rowSource = (wRight * sourceCellCenter[1] +
                wUp * Float32(rowEnd) + wDown * Float32(rowBegin)) / wAll
            colSource = (wRight * sourceCellCenter[2] +
                wUp * Float32(numCols) + wDown * Float32(numCols)) / wAll
            imageOutput[row, col] = computePixel(
                imageInput, rowSource, colSource,
            )
            movements[row, col] = (
                rowSource - Float32(row), colSource - Float32(col),
            )
        end
    end
end

function computeLeftBottomTriangle(
    imageOutput::Matrix{RGB{Float32}},
    movements::Matrix{Tuple{Float32,Float32}},
    ltSourceCellCenter::Tuple{Float32,Float32},
    lbSourceCellCenter::Tuple{Float32,Float32},
    rbSourceCellCenter::Tuple{Float32,Float32},
    ltDestCellCenter::Tuple{Float32,Float32},
    lbDestCellCenter::Tuple{Float32,Float32},
    rbDestCellCenter::Tuple{Float32,Float32},
    imageInput::Matrix{RGB{Float32}},
    )
    @assert ltDestCellCenter[2] == lbDestCellCenter[2]
    @assert lbDestCellCenter[1] == rbDestCellCenter[1]
    rowBegin = Int(ltDestCellCenter[1])
    rowEnd = Int(rbDestCellCenter[1])
    colBegin = Int(ltDestCellCenter[2])
    colEnd = Int(rbDestCellCenter[2])
    for row = rowBegin:rowEnd
        w = (row - rowBegin) / (rowEnd - rowBegin)
        colBeginOfRow = colBegin
        colEndOfRow = Int(floor((1.0 - w) * colBegin + w * colEnd))
        for col = colBeginOfRow:colEndOfRow
            wLeft = computeTriangleArea(
                Float32(rowBegin), Float32(colBegin),
                Float32(rowEnd), Float32(colBegin),
                Float32(row), Float32(col),
            )
            wUp = computeTriangleArea(
                Float32(rowBegin), Float32(colBegin),
                Float32(row), Float32(col),
                Float32(rowEnd), Float32(colEnd),
            )
            wDown = computeTriangleArea(
                Float32(rowEnd), Float32(colBegin),
                Float32(row), Float32(col),
                Float32(rowEnd), Float32(colEnd),
            )
            wAll = wLeft + wUp + wDown
            rowSource = (wLeft * rbSourceCellCenter[1]
                + wUp * lbSourceCellCenter[1] + wDown * ltSourceCellCenter[1]
            ) / wAll
            colSource = (wLeft * rbSourceCellCenter[2]
                + wUp * lbSourceCellCenter[2] + wDown * ltSourceCellCenter[2]
            ) / wAll
            imageOutput[row, col] = computePixel(
                imageInput, rowSource, colSource,
            )
            movements[row, col] = (
                rowSource - Float32(row), colSource - Float32(col),
            )
        end
    end
end

function computeRightTopTriangle(
    imageOutput::Matrix{RGB{Float32}},
    movements::Matrix{Tuple{Float32,Float32}},
    ltSourceCellCenter::Tuple{Float32,Float32},
    rtSourceCellCenter::Tuple{Float32,Float32},
    rbSourceCellCenter::Tuple{Float32,Float32},
    ltDestCellCenter::Tuple{Float32,Float32},
    rtDestCellCenter::Tuple{Float32,Float32},
    rbDestCellCenter::Tuple{Float32,Float32},
    imageInput::Matrix{RGB{Float32}},
    )
    @assert ltDestCellCenter[1] == rtDestCellCenter[1]
    @assert rtDestCellCenter[2] == rbDestCellCenter[2]
    rowBegin = Int(ltDestCellCenter[1])
    rowEnd = Int(rbDestCellCenter[1])
    colBegin = Int(ltDestCellCenter[2])
    colEnd = Int(rbDestCellCenter[2])
    for row = rowBegin:rowEnd
        w = (row - rowBegin) / (rowEnd - rowBegin)
        colBeginOfRow = Int(ceil((1.0 - w) * colBegin + w * colEnd))
        colEndOfRow = colEnd
        for col = colBeginOfRow:colEndOfRow
            wRight = computeTriangleArea(
                Float32(rowBegin), Float32(colEnd),
                Float32(rowEnd), Float32(colEnd),
                Float32(row), Float32(col),
            )
            wUp = computeTriangleArea(
                Float32(rowBegin), Float32(colBegin),
                Float32(rowBegin), Float32(colEnd),
                Float32(row), Float32(col),
            )
            wDown = computeTriangleArea(
                Float32(rowBegin), Float32(colBegin),
                Float32(row), Float32(col),
                Float32(rowEnd), Float32(colEnd),
            )
            wAll = wRight + wUp + wDown
            rowSource = (wRight * ltSourceCellCenter[1]
                + wUp * rbSourceCellCenter[1] + wDown * rtSourceCellCenter[1]
            ) / wAll
            colSource = (wRight * ltSourceCellCenter[2]
                + wUp * rbSourceCellCenter[2] + wDown * rtSourceCellCenter[2]
            ) / wAll
            imageOutput[row, col] = computePixel(
                imageInput, rowSource, colSource,
            )
            movements[row, col] = (
                rowSource - Float32(row), colSource - Float32(col),
            )
        end
    end
end

function distortImage(
    numRowCells::Int,
    numColCells::Int,
    imageInput::Matrix{RGB{Float32}},
    )::Tuple{Matrix{RGB{Float32}},Matrix{Tuple{Float32,Float32}}}
    (numRows, numCols) = size(imageInput)
    @assert numRowCells > 0 && numRowCells * 2 < numRows
    @assert numColCells > 0 && numColCells * 2 < numCols

    # compute cell centers in dest image
    destCellCenters = Matrix{Tuple{Float32,Float32}}(undef,
        (numRowCells, numColCells))
    for cellRow = 1:numRowCells
        rowBegin = (cellRow - 1) * (numRows - 1) ÷ numRowCells + 1
        rowEnd = cellRow * (numRows - 1) ÷ numRowCells + 1
        for cellCol = 1:numColCells
            colBegin = (cellCol - 1) * (numCols - 1) ÷ numColCells + 1
            colEnd = cellCol * (numCols - 1) ÷ numColCells + 1
            destCellCenters[cellRow,cellCol] = (
                (rowBegin + rowEnd) ÷ 2,
                (colBegin + colEnd) ÷ 2,
            )
        end
    end

    # compute cell centers in source image
    sourceCellCenters = Matrix{Tuple{Float32,Float32}}(undef,
        (numRowCells, numColCells))
    for cellRow = 1:numRowCells
        rowBegin = (cellRow - 1) * (numRows - 1) ÷ numRowCells + 1
        rowEnd = cellRow * (numRows - 1) ÷ numRowCells + 1
        for cellCol = 1:numColCells
            colBegin = (cellCol - 1) * (numCols - 1) ÷ numColCells + 1
            colEnd = cellCol * (numCols - 1) ÷ numColCells + 1
            rowPos = rand(0.25:0.001:0.75)
            colPos = rand(0.25:0.001:0.75)
            sourceCellCenters[cellRow,cellCol] = (
                (1 - rowPos) * rowBegin + rowPos * rowEnd,
                (1 - colPos) * colBegin + colPos * colEnd,
            )
        end
    end

    # allocate arrays for results
    imageOutput = Matrix{RGB{Float32}}(undef, (numRows, numCols))
    movements = Matrix{Tuple{Float32,Float32}}(undef, (numRows, numCols))

    # compute the first and last row of dest image
    function firstAndLastRowMap(idx1,idx2)
        for idx = idx1:idx2
            if idx <= numColCells
                cellCol = idx
                colBegin = (cellCol - 1) * (numCols - 1) ÷ numColCells + 1
                colEnd = cellCol * (numCols - 1) ÷ numColCells + 1
                computeFirstRowUpTriangle!(
                    imageOutput,
                    movements,
                    colBegin,
                    colEnd,
                    sourceCellCenters[1, cellCol],
                    destCellCenters[1, cellCol],
                    imageInput,
                )
            elseif idx <= 2 * numColCells
                cellCol = idx - numColCells
                colBegin = (cellCol - 1) * (numCols - 1) ÷ numColCells + 1
                colEnd = cellCol * (numCols - 1) ÷ numColCells + 1
                computeLastRowDownTriangle!(
                    imageOutput,
                    movements,
                    colBegin,
                    colEnd,
                    sourceCellCenters[numRowCells, cellCol],
                    destCellCenters[numRowCells, cellCol],
                    imageInput,
                )
            elseif idx <= 3 * numColCells - 1
                cellCol = idx - 2 * numColCells
                colTop = cellCol * (numCols - 1) ÷ numColCells + 1
                computeFirstRowDownTriangle!(
                    imageOutput,
                    movements,
                    colTop,
                    sourceCellCenters[1, cellCol],
                    destCellCenters[1, cellCol],
                    sourceCellCenters[1, cellCol + 1],
                    destCellCenters[1, cellCol + 1],
                    imageInput,
                )
            else
                cellCol = idx - 3 * numColCells + 1
                colBottom = cellCol * (numCols - 1) ÷ numColCells + 1
                computeLastRowUpTriangle!(
                    imageOutput,
                    movements,
                    colBottom,
                    sourceCellCenters[numRowCells, cellCol],
                    destCellCenters[numRowCells, cellCol],
                    sourceCellCenters[numRowCells, cellCol + 1],
                    destCellCenters[numRowCells, cellCol + 1],
                    imageInput,
                )
            end
        end
    end
    mapOnly(1,4*numColCells-2,1,firstAndLastRowMap,true)

    # compute the first and last column of dest image
    function firstAndLastColMap(idx1,idx2)
        for idx = idx1:idx2
            if idx <= numRowCells
                cellRow = idx
                rowBegin = (cellRow - 1) * (numRows - 1) ÷ numRowCells + 1
                rowEnd = cellRow * (numRows - 1) ÷ numRowCells + 1
                computeFirstColLeftTriangle!(
                    imageOutput,
                    movements,
                    rowBegin,
                    rowEnd,
                    sourceCellCenters[cellRow, 1],
                    destCellCenters[cellRow, 1],
                    imageInput,
                    )
            elseif idx <= 2 * numRowCells
                cellRow = idx - numRowCells
                rowBegin = (cellRow - 1) * (numRows - 1) ÷ numRowCells + 1
                rowEnd = cellRow * (numRows - 1) ÷ numRowCells + 1
                computeLastColRightTriangle!(
                    imageOutput,
                    movements,
                    rowBegin,
                    rowEnd,
                    sourceCellCenters[cellRow, numColCells],
                    destCellCenters[cellRow, numColCells],
                    imageInput,
                )
            elseif idx <= 3 * numRowCells - 1
                cellRow = idx - 2 * numRowCells
                rowLeft = cellRow * (numRows - 1) ÷ numRowCells + 1
                computeFirstColRightTriangle!(
                    imageOutput,
                    movements,
                    rowLeft,
                    sourceCellCenters[cellRow, 1],
                    destCellCenters[cellRow, 1],
                    sourceCellCenters[cellRow + 1, 1],
                    destCellCenters[cellRow + 1, 1],
                    imageInput,
                )
            else
                cellRow = idx - 3 * numRowCells + 1
                rowRight = cellRow * (numRows - 1) ÷ numRowCells + 1
                computeLastColLeftTriangle!(
                    imageOutput,
                    movements,
                    rowRight,
                    sourceCellCenters[cellRow, numColCells],
                    destCellCenters[cellRow, numColCells],
                    sourceCellCenters[cellRow + 1, numColCells],
                    destCellCenters[cellRow + 1, numColCells],
                    imageInput,
                )
            end
        end
    end
    mapOnly(1,4*numRowCells-2,1,firstAndLastColMap,true)

    # compute inner areas of dest image
    function innerAreaMap(idx1, idx2)
        for idx = idx1 : idx2
            cellRow = (idx - 1) ÷ (numColCells - 1) + 1
            cellCol = (idx - 1) % (numColCells - 1) + 1
            ltSourceCellCenter = sourceCellCenters[cellRow, cellCol]
            rtSourceCellCenter = sourceCellCenters[cellRow, cellCol + 1]
            lbSourceCellCenter = sourceCellCenters[cellRow + 1, cellCol]
            rbSourceCellCenter = sourceCellCenters[cellRow + 1, cellCol + 1]
            ltDestCellCenter = destCellCenters[cellRow, cellCol]
            rtDestCellCenter = destCellCenters[cellRow, cellCol + 1]
            lbDestCellCenter = destCellCenters[cellRow + 1, cellCol]
            rbDestCellCenter = destCellCenters[cellRow + 1, cellCol + 1]
            computeLeftBottomTriangle(
                imageOutput,
                movements,
                ltSourceCellCenter,
                lbSourceCellCenter,
                rbSourceCellCenter,
                ltDestCellCenter,
                lbDestCellCenter,
                rbDestCellCenter,
                imageInput,
            )
            computeRightTopTriangle(
                imageOutput,
                movements,
                ltSourceCellCenter,
                rtSourceCellCenter,
                rbSourceCellCenter,
                ltDestCellCenter,
                rtDestCellCenter,
                rbDestCellCenter,
                imageInput,
            )
        end
    end
    mapOnly(1, (numRowCells - 1) * (numColCells - 1), 1, innerAreaMap,true)

    # make and return result
    return (imageOutput, movements)
end

function generateMatchingDatasets(
    dataSetPath::String,
    numSamplesPerVideo::Int,
    numItersPerSample::Int
    )::Vector{Tuple{
        Matrix{RGB{Float32}}, Matrix{RGB{Float32}},
        Matrix{Tuple{Float32,Float32}},
    }}
    samples = Matrix{RGB{Float32}}[]
    for fileName in readdir(dataSetPath)
        if !endswith(fileName, ".mp4")
            continue
        end
        frames = VideoIO.load(joinpath(dataSetPath, fileName))
        numFrames = length(frames)
        append!(samples, [
                imresize(
                    RGB{Float32}.(frames[rand(1:numFrames)]),
                    (384,640),
                ) for idxSample = 1:numSamplesPerVideo
        ])
    end
    result = Vector{Tuple{
        Matrix{RGB{Float32}}, Matrix{RGB{Float32}},
        Matrix{Tuple{Float32,Float32}},
    }}()
    for sample in samples
        for idxIter = 1:numItersPerSample
            (distortedSample, distortion) = distortImage(16, 16, sample)
            push!(result, (
                sample,
                distortedSample,
                distortion,
            ))
        end
    end
    shuffle!(result)
    return result
end

struct WjfMatchingUNet
    inputs::PyObject
    conv1a::PyObject
    conv1b::PyObject
    pool1::PyObject
    conv2a::PyObject
    conv2b::PyObject
    pool2::PyObject
    conv3a::PyObject
    conv3b::PyObject
    pool3::PyObject
    conv4a::PyObject
    conv4b::PyObject
    drop4::PyObject
    pool4::PyObject
    conv5a::PyObject
    conv5b::PyObject
    drop5::PyObject
    up6::PyObject
    merge6::PyObject
    conv6a::PyObject
    conv6b::PyObject
    up7::PyObject
    merge7::PyObject
    conv7a::PyObject
    conv7b::PyObject
    up8::PyObject
    merge8::PyObject
    conv8a::PyObject
    conv8b::PyObject
    up9::PyObject
    merge9::PyObject
    conv9a::PyObject
    conv9b::PyObject
    conv9c::PyObject
    conv10::PyObject
    model::PyObject
end

function createMatchingUNet(inputSize::Tuple{Int,Int,Int})::WjfMatchingUNet
    tf = pyimport("tensorflow")
    keras = tf.keras
    models = keras.models
    layers = keras.layers
    optimizers = keras.optimizers
    inputs = layers.Input(inputSize)
    conv1a = layers.Conv2D(64, 3, activation = "relu", padding = "same",
        kernel_initializer = "he_normal")(inputs)
    conv1b = layers.Conv2D(64, 3, activation = "relu", padding = "same",
        kernel_initializer = "he_normal")(conv1a)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1b)
    conv2a = layers.Conv2D(128, 3, activation = "relu", padding = "same",
        kernel_initializer = "he_normal")(pool1)
    conv2b = layers.Conv2D(128, 3, activation = "relu", padding = "same",
        kernel_initializer = "he_normal")(conv2a)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2b)
    conv3a = layers.Conv2D(256, 3, activation = "relu", padding = "same",
        kernel_initializer = "he_normal")(pool2)
    conv3b = layers.Conv2D(256, 3, activation = "relu", padding = "same",
        kernel_initializer = "he_normal")(conv3a)
    pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3b)
    conv4a = layers.Conv2D(512, 3, activation = "relu", padding = "same",
        kernel_initializer = "he_normal")(pool3)
    conv4b = layers.Conv2D(512, 3, activation = "relu", padding = "same",
        kernel_initializer = "he_normal")(conv4a)
    drop4 = layers.Dropout(0.5)(conv4b)
    pool4 = layers.MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5a = layers.Conv2D(1024, 3, activation = "relu", padding = "same",
        kernel_initializer = "he_normal")(pool4)
    conv5b = layers.Conv2D(1024, 3, activation = "relu", padding = "same",
        kernel_initializer = "he_normal")(conv5a)
    drop5 = layers.Dropout(0.5)(conv5b)

    up6 = layers.Conv2D(512, 2, activation = "relu", padding = "same",
        kernel_initializer = "he_normal")(
        layers.UpSampling2D(size = (2,2))(drop5))
    merge6 = layers.concatenate([drop4,up6], axis = 3)
    conv6a = layers.Conv2D(512, 3, activation = "relu", padding = "same",
        kernel_initializer = "he_normal")(merge6)
    conv6b = layers.Conv2D(512, 3, activation = "relu", padding = "same",
        kernel_initializer = "he_normal")(conv6a)

    up7 = layers.Conv2D(256, 2, activation = "relu", padding = "same",
        kernel_initializer = "he_normal")(
        layers.UpSampling2D(size = (2,2))(conv6b))
    merge7 = layers.concatenate([conv3b,up7], axis = 3)
    conv7a = layers.Conv2D(256, 3, activation = "relu", padding = "same",
        kernel_initializer = "he_normal")(merge7)
    conv7b = layers.Conv2D(256, 3, activation = "relu", padding = "same",
        kernel_initializer = "he_normal")(conv7a)

    up8 = layers.Conv2D(128, 2, activation = "relu", padding = "same",
        kernel_initializer = "he_normal")(
        layers.UpSampling2D(size = (2,2))(conv7b))
    merge8 = layers.concatenate([conv2b,up8], axis = 3)
    conv8a = layers.Conv2D(128, 3, activation = "relu", padding = "same",
        kernel_initializer = "he_normal")(merge8)
    conv8b = layers.Conv2D(128, 3, activation = "relu", padding = "same",
        kernel_initializer = "he_normal")(conv8a)

    up9 = layers.Conv2D(64, 2, activation = "relu", padding = "same",
        kernel_initializer = "he_normal")(
        layers.UpSampling2D(size = (2,2))(conv8b))
    merge9 = layers.concatenate([conv1b,up9], axis = 3)
    conv9a = layers.Conv2D(64, 3, activation = "relu", padding = "same",
        kernel_initializer = "he_normal")(merge9)
    conv9b = layers.Conv2D(64, 3, activation = "relu", padding = "same",
        kernel_initializer = "he_normal")(conv9a)
    conv9c = layers.Conv2D(16, 3, activation = "relu", padding = "same",
        kernel_initializer = "he_normal")(conv9b)

    conv10 = layers.Conv2D(2, 1)(conv9c)

    model = models.Model(inputs = inputs, outputs = conv10)

    model.compile(optimizer = optimizers.Adam(learning_rate = 1e-4),
        loss = keras.losses.Huber(),
        metrics = ["accuracy"])

    return WjfMatchingUNet(
        inputs,
        conv1a, conv1b, pool1,
        conv2a, conv2b, pool2,
        conv3a, conv3b, pool3,
        conv4a, conv4b, drop4, pool4,
        conv5a, conv5b, drop5,
        up6, merge6, conv6a, conv6b,
        up7, merge7, conv7a, conv7b,
        up8, merge8, conv8a, conv8b,
        up9, merge9, conv9a, conv9b, conv9c,
        conv10,
        model,
    )
end

function trainMatchingUNet(
    dataSetPath::String,
    numSamplesPerVideo::Int,
    numItersPerSample::Int,
    batchSize::Int,
    epochs::Int,
    rounds::Int,
    )
    println("sampling data...")
    dataSet = generateMatchingDatasets(dataSetPath, numSamplesPerVideo,
        numItersPerSample)
    (h,w) = size(dataSet[1][1])
    n = length(dataSet)
    println("n = $n, h = $h, w = $w")
    xs = Array{Float32,4}(undef, (n, h, w, 6))
    ys = Array{Float32,4}(undef, (n, h, w, 2))
    for idx = 1:n
        (x1, x2, y) = dataSet[idx]
        for row = 1:h
            for col = 1:w
                rgb1 = x1[row, col]
                rgb2 = x2[row, col]
                uv = y[row, col]
                xs[idx, row, col, 1] = rgb1.r
                xs[idx, row, col, 2] = rgb1.g
                xs[idx, row, col, 3] = rgb1.b
                xs[idx, row, col, 4] = rgb2.r
                xs[idx, row, col, 5] = rgb2.g
                xs[idx, row, col, 6] = rgb2.b
                ys[idx, row, col, 1] = uv[1]
                ys[idx, row, col, 2] = uv[2]
            end
        end
    end
    dataSet = nothing
    GC.gc()
    println("creating unet...")
    unet = createMatchingUNet((h, w, 6))
    println("training unet...")
    for round = 1:rounds
        unet.model.fit(xs, ys, epochs=epochs, batch_size=10, verbose=1)
        unet.model.save("matchin_unet_model_$round")
    end
end

end # module
