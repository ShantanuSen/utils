def calcFor2DConvMaxPool(inputShape, outputShape, kernelSize, padding, dilation, stride):

    """
    Calculate shape after 2D convolution or 2D max pooling operation.

    Returns
    ---------
    Tuple of the desired solution.
    
    Parameters
    ----------
    inputShape: "solve" or tuple
    outputShape: "solve" or tuple
    kernelSize: "solve" or tuple
    padding: "solve" or tuple
    dilation: "solve" or tuple
    stride: "solve" or tuple

    Example
    --------
    print(calcFor2DConvMaxPool((32,32), (28,28), (5,5), (0,0), (1,1), "solve"))

    """
    
    
    outputHeight, inputHeight, paddingHeight, dilationHeight, kernelHeight, strideHeight = sympy.symbols('outputHeight inputHeight paddingHeight dilationHeight kernelHeight strideHeight')
    outputWidth, inputWidth, paddingWidth, dilationWidth, kernelWidth, strideWidth = sympy.symbols('outputWidth inputWidth paddingWidth dilationWidth kernelWidth strideWidth')
    equationH = Eq(outputHeight, (((inputHeight + 2*paddingHeight - dilationHeight*(kernelHeight-1)-1)/strideHeight)+1))
    equationW = Eq(outputWidth, (((inputWidth + 2*paddingWidth - dilationWidth*(kernelWidth-1)-1)/strideWidth)+1))
    
    if(outputShape=="solve"):

        subsH = equationH.subs({inputHeight:inputShape[0], kernelHeight:kernelSize[0], paddingHeight:padding[0], dilationHeight:dilation[0], strideHeight:stride[0]})
        subsW = equationW.subs({inputWidth:inputShape[1], kernelWidth:kernelSize[1], paddingWidth:padding[1], dilationWidth:dilation[1], strideWidth:stride[1]})

        height = solve(subsH, outputHeight)
        width = solve(subsW, outputWidth)

        outputshape = (math.floor(height[0]), math.floor(width[0]))

        return outputshape

    elif(inputShape=="solve"):

        subsH = equationH.subs({outputHeight:outputShape[0], kernelHeight:kernelSize[0], paddingHeight:padding[0], dilationHeight:dilation[0], strideHeight:stride[0]})
        subsW = equationW.subs({outputWidth:outputShape[1], kernelWidth:kernelSize[1], paddingWidth:padding[1], dilationWidth:dilation[1], strideWidth:stride[1]})

        height = solve(subsH, inputHeight)
        width = solve(subsW, inputWidth)

        inputshape = (math.floor(height[0]), math.floor(width[0]))

        return inputshape

    elif(kernelSize=="solve"):

        subsH = equationH.subs({outputHeight:outputShape[0], inputHeight:inputShape[0], paddingHeight:padding[0], dilationHeight:dilation[0], strideHeight:stride[0]})
        subsW = equationW.subs({outputWidth:outputShape[1], inputWidth:inputShape[1], paddingWidth:padding[1], dilationWidth:dilation[1], strideWidth:stride[1]})

        height = solve(subsH, kernelHeight)
        width = solve(subsW, kernelWidth)

        kernelshape = (math.floor(height[0]), math.floor(width[0]))

        return kernelshape

    elif(padding=="solve"):

        subsH = equationH.subs({outputHeight:outputShape[0], inputHeight:inputShape[0], kernelHeight:kernelSize[0], dilationHeight:dilation[0], strideHeight:stride[0]})
        subsW = equationW.subs({outputWidth:outputShape[1], inputWidth:inputShape[1], kernelWidth:kernelSize[1], dilationWidth:dilation[1], strideWidth:stride[1]})

        height = solve(subsH, paddingHeight)
        width = solve(subsW, paddingWidth)

        paddingshape = (math.floor(height[0]), math.floor(width[0]))

        return paddingshape

    elif(dilation=="solve"):

        subsH = equationH.subs({outputHeight:outputShape[0], inputHeight:inputShape[0], kernelHeight:kernelSize[0], paddingHeight:padding[0], strideHeight:stride[0]})
        subsW = equationW.subs({outputWidth:outputShape[1], inputWidth:inputShape[1], kernelWidth:kernelSize[1], paddingWidth:padding[1], strideWidth:stride[1]})

        height = solve(subsH, dilationHeight)
        width = solve(subsW, dilationWidth)

        dilationshape = (math.floor(height[0]), math.floor(width[0]))

        return dilationshape

    elif(stride=="solve"):

        subsH = equationH.subs({outputHeight:outputShape[0], inputHeight:inputShape[0],kernelHeight:kernelSize[0], paddingHeight:padding[0], dilationHeight:dilation[0]})
        subsW = equationW.subs({outputWidth:outputShape[1], inputWidth:inputShape[1], kernelWidth:kernelSize[1], paddingWidth:padding[1], dilationWidth:dilation[1]})

        height = solve(subsH, strideHeight)
        width = solve(subsW, strideWidth)

        strideshape = (math.floor(height[0]), math.floor(width[0]))

        return strideshape
