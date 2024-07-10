module MarchalConjecture

    using RadiiPolynomial

include("symmetry.jl")
    export EvenCos, OddCos, EvenSin, OddSin, EvenFourier, project_sym!, norm1_sym, opnorm1_sym

include("marchal.jl")
    export F_marchal!, DF_marchal!

#

function grid2cheb(x_grid::Vector{<:Sequence}, N)
    x_fft = [x_grid[end:-1:1] ; x_grid[2:end-1]]
    return Sequence(space(x_grid[1]),
        [ifft!(getindex.(x_fft, i), Chebyshev(N)) for i ∈ indices(space(x_grid[1]))])
end

function grid2cheb(A_grid::Vector{<:LinearOperator}, N)
    A_fft = [A_grid[end:-1:1] ; A_grid[2:end-1]]
    return LinearOperator(domain(A_fft[1]), codomain(A_fft[1]),
        [ifft!(getindex.(A_fft, i, j), Chebyshev(N)) for i ∈ indices(codomain(A_fft[1])), j ∈ indices(domain(A_fft[1]))])
end

function cheb2grid(x_cheb, N_fft)
    npts = N_fft ÷ 2 + 1
    x_fft = fft.(x_cheb, N_fft)
    return [getindex.(x_fft, i) for i ∈ npts:-1:1]
end

function get_tail(x, K)
    y = copy(x)
    y[(:,-K:K)] .= 0
    return y
end

    export grid2cheb, cheb2grid, get_tail

end
