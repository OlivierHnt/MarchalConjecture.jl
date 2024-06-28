function F_marchal!(F, x, Ω)
    a = x[1]
    β = x[2]
    α = x[3]
    u, v, w = component(x, 4), component(x, 5), component(x, 6)

    ζ = ExactReal(2) * convert(real(eltype(x)), π) / ExactReal(3)

    ℰ₀ = Evaluation(0)
    𝒮  = Shift(ExactReal(2) * ζ)
    𝒮² = Shift(ExactReal(4) * ζ)

    # amplitude equation

    F[1] = ℰ₀ * component(u, 3) - ExactReal(1)

    # zero average

    F[2] = component(u, 1)[0]

    # initial condition for the polynomial embedding

    M₁u_x = component(u, 1) - 𝒮 * component(u, 1)
    M₁u_y = component(u, 2) - 𝒮 * component(u, 2)
    M₁u_z = component(u, 3) - 𝒮 * component(u, 3)

    ℰ₀M₁u_x = ℰ₀ * M₁u_x
    ℰ₀M₁u_y = ℰ₀ * M₁u_y
    ℰ₀M₁u_z = ℰ₀ * M₁u_z

    ℰ₀w = ℰ₀ * w
    ℰ₀w² = ℰ₀w * ℰ₀w

    F[3] = ℰ₀w² * (ℰ₀M₁u_x * ℰ₀M₁u_x + ℰ₀M₁u_y * ℰ₀M₁u_y + a * ℰ₀M₁u_z * ℰ₀M₁u_z) - 1

    #

    M₂u_x = component(u, 1) - 𝒮² * component(u, 1)
    M₂u_y = component(u, 2) - 𝒮² * component(u, 2)
    M₂u_z = component(u, 3) - 𝒮² * component(u, 3)

    w³ = w * w * w
    Rw³ = Sequence(space(w³), reverse(coefficients(w³)))

    project!(component(component(F, 4), 1), β - Ω^2 * component(u, 1) + ExactReal(2) * Ω * component(v, 2) + differentiate(component(v, 1)) + w³ * M₁u_x + Rw³ * M₂u_x)
    project!(component(component(F, 4), 2),   - Ω^2 * component(u, 2) - ExactReal(2) * Ω * component(v, 1) + differentiate(component(v, 2)) + w³ * M₁u_y + Rw³ * M₂u_y)
    project!(component(component(F, 4), 3),                                                                  differentiate(component(v, 3)) + w³ * M₁u_z + Rw³ * M₂u_z)

    #

    project!(component(F, 5), differentiate(u) - v)

    #

    M₁v_x = component(v, 1) - 𝒮 * component(v, 1)
    M₁v_y = component(v, 2) - 𝒮 * component(v, 2)
    M₁v_z = component(v, 3) - 𝒮 * component(v, 3)

    project!(component(F, 6), α + differentiate(w) + w * w * w * (M₁u_x * M₁v_x + M₁u_y * M₁v_y + a * M₁u_z * M₁v_z))

    #

    return F
end

function DF_marchal!(DF, x, Ω)
    a = x[1]
    u, v, w = component(x, 4), component(x, 5), component(x, 6)

    ζ = ExactReal(2) * convert(real(eltype(x)), π) / ExactReal(3)

    ℰ₀  = Evaluation(ExactReal(0))
    ℰ₀𝒮 = Evaluation(ExactReal(2) * ζ)
    𝒮  = Shift(ExactReal(2) * ζ)
    𝒮² = Shift(ExactReal(4) * ζ)

    CoefType = eltype(DF)

    DF .= zero(CoefType)

    # derivative amplitude equation

    project!(component(component(DF, 1, 4), 3), ℰ₀)

    # derivative zero average

    component(component(DF, 2, 4), 1)[1,0] = ExactReal(1)

    # derivative initial condition for the polynomial embedding

    M₁u_x = component(u, 1) - 𝒮 * component(u, 1)
    M₁u_y = component(u, 2) - 𝒮 * component(u, 2)
    M₁u_z = component(u, 3) - 𝒮 * component(u, 3)

    ℰ₀M₁u_x = ℰ₀ * M₁u_x
    ℰ₀M₁u_y = ℰ₀ * M₁u_y
    ℰ₀M₁u_z = ℰ₀ * M₁u_z

    ℰ₀w = ℰ₀ * w
    ℰ₀w² = ℰ₀w * ℰ₀w

    sub!(component(component(DF, 3, 4), 1),  ℰ₀, ℰ₀𝒮) .*=     ExactReal(2) * ℰ₀w² * ℰ₀M₁u_x
    sub!(component(component(DF, 3, 4), 2),  ℰ₀, ℰ₀𝒮) .*=     ExactReal(2) * ℰ₀w² * ℰ₀M₁u_y
    sub!(component(component(DF, 3, 4), 3),  ℰ₀, ℰ₀𝒮) .*= a * ExactReal(2) * ℰ₀w² * ℰ₀M₁u_z

    project!(component(DF, 3, 6), ℰ₀) .*= ExactReal(2) * ℰ₀w * (ℰ₀M₁u_x * ℰ₀M₁u_x + ℰ₀M₁u_y * ℰ₀M₁u_y + a * ℰ₀M₁u_z * ℰ₀M₁u_z)

    component(DF, 3, 1)[1,1] = ℰ₀w² * ℰ₀M₁u_z * ℰ₀M₁u_z

    #

    M₂u_x = component(u, 1) - 𝒮² * component(u, 1)
    M₂u_y = component(u, 2) - 𝒮² * component(u, 2)
    M₂u_z = component(u, 3) - 𝒮² * component(u, 3)

    w² = w * w
    w³ = w² * w
    Rw² = Sequence(space(w²), reverse(coefficients(w²)))
    Rw³ = Sequence(space(w³), reverse(coefficients(w³)))

    project!(component(component(DF, 4, 4), 1, 1), UniformScaling(-Ω^2))
    project!(component(component(DF, 4, 5), 1, 1), Derivative(1))
    project!(component(component(DF, 4, 5), 1, 2), ExactReal(2) * UniformScaling(Ω))

    project!(component(component(DF, 4, 4), 2, 2), UniformScaling(-Ω^2))
    project!(component(component(DF, 4, 5), 2, 1), ExactReal(-2) * UniformScaling(Ω))
    project!(component(component(DF, 4, 5), 2, 2), Derivative(1))

    project!(component(component(DF, 4, 5), 3, 3), Derivative(1))

    _dom_ = domain(component(component(DF, 4, 4), 1, 1))
    mul!(component(component(DF, 4, 4), 1, 1), Multiplication(w³),  UniformScaling(ExactReal(1)) - project(𝒮, _dom_, _dom_, CoefType), ExactReal(true), ExactReal(true))
    mul!(component(component(DF, 4, 4), 1, 1), Multiplication(Rw³), UniformScaling(ExactReal(1)) - project(𝒮², _dom_, _dom_, CoefType), ExactReal(true), ExactReal(true))

    _dom_ = domain(component(component(DF, 4, 4), 2, 2))
    mul!(component(component(DF, 4, 4), 2, 2), Multiplication(w³),  UniformScaling(ExactReal(1)) - project(𝒮, _dom_, _dom_, CoefType), ExactReal(true), ExactReal(true))
    mul!(component(component(DF, 4, 4), 2, 2), Multiplication(Rw³), UniformScaling(ExactReal(1)) - project(𝒮², _dom_, _dom_, CoefType), ExactReal(true), ExactReal(true))

    _dom_ = domain(component(component(DF, 4, 4), 3, 3))
    mul!(component(component(DF, 4, 4), 3, 3), Multiplication(w³),  UniformScaling(ExactReal(1)) - project(𝒮, _dom_, _dom_, CoefType), ExactReal(true), ExactReal(true))
    mul!(component(component(DF, 4, 4), 3, 3), Multiplication(Rw³), UniformScaling(ExactReal(1)) - project(𝒮², _dom_, _dom_, CoefType), ExactReal(true), ExactReal(true))

    project!(component(component(DF, 4, 6), 1), Multiplication(ExactReal(3) * w² * M₁u_x))
    R = zero.(component(component(DF, 4, 6), 1)); project!(R, I); reverse!(coefficients(R); dims = 2)
    mul!(component(component(DF, 4, 6), 1),     Multiplication(ExactReal(3) * Rw² * M₂u_x), R, ExactReal(true), ExactReal(true))
    project!(component(component(DF, 4, 6), 2), Multiplication(ExactReal(3) * w² * M₁u_y))
    R = zero.(component(component(DF, 4, 6), 2)); project!(R, I); reverse!(coefficients(R); dims = 2)
    mul!(component(component(DF, 4, 6), 2),     Multiplication(ExactReal(3) * Rw² * M₂u_y), R, ExactReal(true), ExactReal(true))
    project!(component(component(DF, 4, 6), 3), Multiplication(ExactReal(3) * w² * M₁u_z))
    R = zero.(component(component(DF, 4, 6), 3)); project!(R, I); reverse!(coefficients(R); dims = 2)
    mul!(component(component(DF, 4, 6), 3),     Multiplication(ExactReal(3) * Rw² * M₂u_z), R, ExactReal(true), ExactReal(true))

    component(component(DF, 4, 2), 1)[0,1] = ExactReal(1)

    #

    project!(component(DF, 5, 4), Derivative(1))
    project!(component(DF, 5, 5), UniformScaling(ExactReal(-1)))

    #

    M₁v_x = component(v, 1) - 𝒮 * component(v, 1)
    M₁v_y = component(v, 2) - 𝒮 * component(v, 2)
    M₁v_z = component(v, 3) - 𝒮 * component(v, 3)

    _dom_ = domain(component(component(DF, 6, 4), 1))
    mul!(component(component(DF, 6, 4), 1), Multiplication(    w³ * M₁v_x), UniformScaling(ExactReal(1)) - project(𝒮, _dom_, _dom_, CoefType))
    _dom_ = domain(component(component(DF, 6, 4), 2))
    mul!(component(component(DF, 6, 4), 2), Multiplication(    w³ * M₁v_y), UniformScaling(ExactReal(1)) - project(𝒮, _dom_, _dom_, CoefType))
    _dom_ = domain(component(component(DF, 6, 4), 3))
    mul!(component(component(DF, 6, 4), 3), Multiplication(a * w³ * M₁v_z), UniformScaling(ExactReal(1)) - project(𝒮, _dom_, _dom_, CoefType))

    _dom_ = domain(component(component(DF, 6, 5), 1))
    mul!(component(component(DF, 6, 5), 1), Multiplication(    w³ * M₁u_x), UniformScaling(ExactReal(1)) - project(𝒮, _dom_, _dom_, CoefType))
    _dom_ = domain(component(component(DF, 6, 5), 2))
    mul!(component(component(DF, 6, 5), 2), Multiplication(    w³ * M₁u_y), UniformScaling(ExactReal(1)) - project(𝒮, _dom_, _dom_, CoefType))
    _dom_ = domain(component(component(DF, 6, 5), 3))
    mul!(component(component(DF, 6, 5), 3), Multiplication(a * w³ * M₁u_z), UniformScaling(ExactReal(1)) - project(𝒮, _dom_, _dom_, CoefType))

    add!(component(DF, 6, 6), Derivative(1), Multiplication(ExactReal(3) * w² * (M₁u_x * M₁v_x + M₁u_y * M₁v_y + a * M₁u_z * M₁v_z)))

    component(DF, 6, 3)[0,1] = ExactReal(1)

    project!(component(DF, 6, 1), w³ * M₁u_z * M₁v_z)

    #

    return DF
end
