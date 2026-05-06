# -*- coding: utf-8 -*-
"""
modelo_fisico_a320.py

Este módulo contiene la formulación matemática precisa de la dinámica de vuelo
del Airbus A320 para la fase de ascenso. Implementa el cálculo de la función
de costo (Phi) que penaliza el consumo de combustible y el tiempo de vuelo.

El código está optimizado con Numba (JIT Compilation) para permitir una
evaluación ultrarrápida evita lentitud python por su naturaleza de lenguaje interpretado
requisito fundamental para los algoritmos de optimización
clásica y muestreo de espacios multidimensionales.

Autor: Marlio Erazo (Adaptado de las ecuaciones oficiales de Airbus)
"""

import numpy as np
import numba
from math import sin, tan, sqrt, pow, asin, atanh, log, exp, pi, nan

# =============================================================================
# CONSTANTES FÍSICAS Y AERODINÁMICAS (Airbus A320)
# =============================================================================
Cx_0 = 0.014
k = 0.09
Cz_max = 0.7
S_REF = 120.0
η = 0.06 / 3600.0
Zp_I = 10000.0 * 0.3048
Zp_F = 36000.0 * 0.3048
m_I = 60000.0
CAS_I = 250.0 * 0.5144444444444445
VMO = 350.0 * 0.5144444444444445
MMO = 0.82
M_CRZ = 0.80
L = 400000.0
s_F = L
Vz_min = 1.52400
g_0 = 9.80665
CI = 30.0 / 60.0

m_0 = m_I
t_0 = 0.0
s_0 = 0.0
λ_0 = 1.0

Ts_0 = 288.15
ρ_0 = 1.225
L_z = -0.0065
R = 287.05287
α_0 = -g_0 / (R * L_z)

# =============================================================================
# RELACIONES ATMOSFÉRICAS Y AERODINÁMICAS (Compilación JIT)
# =============================================================================

@numba.jit(nopython=True)
def Zp(i, N):
    return Zp_I + i * (Zp_F - Zp_I) / (N-1)

@numba.jit(nopython=True)
def F_N_MCL(i, N):
    return 140000.0 - 2.53 * (Zp(i, N) / 0.3048)

@numba.jit(nopython=True)
def rho(i, N):
    return ρ_0 * ((Ts_0 + L_z * Zp(i, N)) / Ts_0)**(α_0 - 1.0)

@numba.jit(nopython=True)
def Mach(l, v, N):
    return v[l] / sqrt(1.4 * R * (Ts_0 + L_z * Zp(l, N)))

@numba.jit(nopython=True)
def calc_CAS(l, v, N):
    base = 1.0 + (v[l]**2 / (7.0 * R * (Ts_0 + L_z * Zp(l, N))))
    if base < 0: return nan
    term1 = (Ts_0 / (Ts_0 + L_z * Zp(l, N)))**(-α_0)
    arg = (7.0 * R * Ts_0) * ((term1 * (pow(base, 3.5) - 1.0) + 1.0)**(1.0 / 3.5) - 1.0)
    if arg < 0: return nan
    return sqrt(arg)

@numba.jit(nopython=True)
def variables_iniciales(N):
    zp_i = Zp(0, N)
    term1 = ((Ts_0 + L_z * zp_i) / Ts_0)**(-α_0)
    term2 = (1.0 + CAS_I**2 / (7.0 * R * Ts_0))**3.5 - 1.0
    v_0 = sqrt(7.0 * R * (Ts_0 + L_z * zp_i) * ((term1 * term2 + 1.0)**(1.0 / 3.5) - 1.0))

    rho_0 = rho(0, N)
    Cz_0 = m_0 * g_0 / (0.5 * rho_0 * v_0**2 * S_REF)
    fn_0 = F_N_MCL(0, N)

    arg_asin = (fn_0 - 0.5 * rho_0 * v_0**2 * S_REF * (Cx_0 + k * Cz_0)) / (m_0 * g_0)
    if arg_asin < -1.0 or arg_asin > 1.0: return nan, nan, nan, nan, nan

    γ_0 = asin(arg_asin)
    rho_F = ρ_0 * ((Ts_0 + L_z * Zp_F) / Ts_0)**(α_0 - 1.0)
    v_F = M_CRZ * sqrt(1.4 * R * (Ts_0 + L_z * Zp_F))

    return v_0, Cz_0, γ_0, rho_F, v_F

# =============================================================================
# EVALUACIÓN DE LA DINÁMICA DE VUELO
# =============================================================================

@numba.jit(nopython=True)
def _evaluar(x):
    N_vars = len(x)
    N = (N_vars // 2) + 1

    x1 = x[:N_vars//2]
    x2 = x[N_vars//2:]

    v = np.zeros(N, dtype=np.float64)
    γ = np.zeros(N, dtype=np.float64)
    m = np.zeros(N, dtype=np.float64)
    s = np.zeros(N, dtype=np.float64)
    t = np.zeros(N, dtype=np.float64)
    λ = np.zeros(N, dtype=np.float64)
    Cz = np.zeros(N, dtype=np.float64)

    v_0, Cz_0, γ_0, rho_F, v_F = variables_iniciales(N)
    if np.isnan(v_0): return nan

    v[0], γ[0], m[0], s[0], t[0], λ[0], Cz[0] = v_0, γ_0, m_0, s_0, t_0, λ_0, Cz_0

    for i in range(len(x1)):
        v[i+1] = x1[i]
        γ[i+1] = x2[i] * pi / 180.0

    for i in range(N - 1):
        if v[i+1] * sin(γ[i+1]) < Vz_min: return nan
        cas_i1 = calc_CAS(i+1, v, N)
        if np.isnan(cas_i1) or cas_i1 > VMO: return nan

        zp_i, zp_ip1 = Zp(i, N), Zp(i+1, N)
        rho_i, rho_ip1 = rho(i, N), rho(i+1, N)
        fn_i, fn_ip1 = F_N_MCL(i, N), F_N_MCL(i+1, N)

        if sin(γ[i]) == 0 or tan(γ[i]) == 0 or tan(γ[i+1]) == 0 or v[i] == 0 or fn_ip1 == 0: return nan

        # Cálculo analítico de masa m[i+1]
        A = (v[i+1] - v[i]) / (zp_ip1 - zp_i)
        L_term = (-g_0 / v[i+1] + (λ[i] * fn_i) / (m[i] * v[i] * sin(γ[i]))
                  - (0.5 * rho_i * v[i] * S_REF * (Cx_0 + k * Cz[i]**2)) / (m[i] * sin(γ[i])) - g_0 / v[i])
        H = (4.0 * sin(γ[i+1]) / (rho_ip1 * S_REF)) * ((γ[i+1] - γ[i]) / (zp_ip1 - zp_i) +
             g_0 / (2.0 * v[i+1]**2 * tan(γ[i+1])) - (rho_i * S_REF * Cz[i]) / (4.0 * m[i] * sin(γ[i])) +
             g_0 / (2.0 * v[i]**2 * tan(γ[i])))
        I = (-2.0 * v[i+1] * sin(γ[i+1]) / (η * fn_ip1)) * (1.0 / (zp_ip1 - zp_i))
        J = (2.0 * v[i+1] * sin(γ[i+1]) / (η * fn_ip1)) * (m[i] / (zp_ip1 - zp_i)) - (v[i+1] * sin(γ[i+1]) * λ[i] * fn_i) / (fn_ip1 * v[i] * sin(γ[i]))

        inner_sqrt = (A**2 * v[i+1]**2 * sin(γ[i+1])**2 - A * fn_ip1 * I * v[i+1] * sin(γ[i+1]) -
                      A * L_term * v[i+1]**2 * sin(γ[i+1])**2 - 0.25 * Cx_0 * H**2 * S_REF**2 * k * rho_ip1**2 * v[i+1]**4 +
                      0.25 * fn_ip1**2 * I**2 + 0.5 * fn_ip1 * H**2 * J * S_REF * k * rho_ip1 * v[i+1]**2 +
                      0.5 * fn_ip1 * I * L_term * v[i+1] * sin(γ[i+1]) + 0.25 * L_term**2 * v[i+1]**2 * sin(γ[i+1])**2)
        if inner_sqrt < 0: return nan

        num1 = -2.0 * A * v[i+1] * sin(γ[i+1]) + fn_ip1 * I + L_term * v[i+1] * sin(γ[i+1])
        den = H**2 * S_REF * k * rho_ip1 * v[i+1]**2
        if den == 0: return nan
        m[i+1] = (num1 + 2.0 * sqrt(inner_sqrt)) / den

        # Actualización de estados
        term_Cz = ((2.0 * γ[i+1] - 2.0 * γ[i]) / (zp_ip1 - zp_i) - (rho_i * S_REF * Cz[i]) / (2.0 * m[i] * sin(γ[i]))
                   + g_0 / (v[i+1]**2 * tan(γ[i+1])) + g_0 / (v[i]**2 * tan(γ[i])))
        Cz[i+1] = (2.0 * m[i+1] * sin(γ[i+1]) * term_Cz) / (rho_ip1 * S_REF)

        t1 = -2.0 * (v[i+1] * sin(γ[i+1])) / (η * fn_ip1)
        t2 = (m[i+1] - m[i]) / (zp_ip1 - zp_i)
        t3 = (v[i+1] * sin(γ[i+1]) * λ[i] * fn_i) / (fn_ip1 * v[i] * sin(γ[i]))
        λ[i+1] = (t1 * t2) - t3

        s[i+1] = s[i] + 0.5 * ((zp_ip1 - zp_i) / tan(γ[i+1]) + (zp_ip1 - zp_i) / tan(γ[i]))
        t[i+1] = t[i] + 0.5 * ((zp_ip1 - zp_i) / (v[i+1] * sin(γ[i+1])) + (zp_ip1 - zp_i) / (v[i] * sin(γ[i])))

        if not (0.0 <= λ[i+1] <= 1.0 and Cz[i+1] <= Cz_max and Mach(i+1, v, N) <= MMO):
            return nan

    # Fase de aceleración y cálculo del costo final
    N_idx = N - 1
    fn_final = F_N_MCL(N_idx, N)

    A = (-rho_F * S_REF * Cx_0) / (2.0 * m[N_idx]) - (6.0 * k * m[N_idx] * g_0**2) / (rho_F * S_REF * v[N_idx]**4)
    B = (16.0 * k * m[N_idx] * g_0**2) / (rho_F * S_REF * v[N_idx]**3)
    C = (fn_final / m[N_idx]) - (12.0 * k * m[N_idx] * g_0**2) / (rho_F * S_REF * v[N_idx]**2)

    inner_sqrt_D = B**2 - 4.0 * A * C
    if inner_sqrt_D < 0 or A == 0: return nan
    D = sqrt(inner_sqrt_D)

    arg1, arg2 = (2.0 * A * v[N_idx] + B) / D, (2.0 * A * v_F + B) / D
    if abs(arg1) >= 1.0 or abs(arg2) >= 1.0: return nan

    t_B = t[N_idx] + (2.0 / D) * (atanh(arg1) - atanh(arg2))
    m_B = m[N_idx] - η * λ[N_idx] * fn_final * (t_B - t[N_idx])

    log_arg = (D - 2.0 * A * v_F - B) / (D - 2.0 * A * v[N_idx] - B)
    if log_arg <= 0: return nan
    s_B = s[N_idx] + (1.0 / A) * log(log_arg) - ((B + D) / (2.0 * A)) * (t_B - t[N_idx])

    if s[N_idx] <= s_B <= s_F:
        m_F_val = m_B * exp((-2.0 * η * g_0 * sqrt(k * Cx_0) / v_F) * (s_F - s_B))
        return -m_F_val + CI * (t_B - s_B / v_F)
    return nan

# =============================================================================
# FUNCIÓN OBJETIVO
# =============================================================================

def phi(*x):
    vector = np.array(x, dtype=np.float64)
    resultado = _evaluar(vector)
    return resultado

# =============================================================================
# PRUEBA DE EVALUACIÓN
# =============================================================================
if __name__ == "__main__":
    costo = phi(  177.17234635518253, 196.03319600181317, 200.65459372600878, 206.68255806221566, 209.0506589889045,
                  210.50450648077106, 213.65030977646416, 216.09950125282126, 217.04499123681467, 220.03202540995676,
                  218.9015091411552, 220.2999660761746, 220.55789764141423, 220.83372237346668, 221.68671736701458,
                  224.5199690356859, 225.48148864840732, 229.68872267591564, 230.97453486093167, 233.56701522113866,
                  235.29248780427906, 236.5704633362147, 238.42269887524662, 239.370230579441, 240.8642305177853,
                  239.865613664207, 241.7314845462339, 242.32369656682977, 242.97490664995496, 243.36300881336348,
                  241.55821513663375, 243.45553092387394, 243.97401753557335, 244.34636548797147, 245.23139090059976,
                  244.92089957056425, 243.57971585445824, 242.16334528609423, 243.70589849443616, 243.28777946699927,
                  241.068814032494, 240.72201097523833, 241.48730904684032, 241.00660642412356, 242.5654668527609,
                  243.08514135418238, 241.7366071095953, 240.5277984279037, 239.95646748651035, 238.6379773483245,
                  226.10177611988502, 210.76486837166163, 1.0001141399211972, 1.197182916491327, 1.578829452795265,
                  1.7333753094497037, 1.8153302799981192, 1.7688447637911269, 1.6686604107940006, 1.6242013729214353,
                  1.5983302545311657, 1.6923302043272663, 1.7627042720080337, 1.7818201540097638, 1.900532676442694,
                  1.9476511842240178, 1.955143407447494, 1.9517333531970542, 1.8969847237954145, 1.8359359286158323,
                  1.7537271361670737, 1.7114608615743192, 1.6774080906897169, 1.64372800348447, 1.602359812529306,
                  1.5403799695096723, 1.50464380002518, 1.521010744895583, 1.5190379676204797, 1.502794467952361,
                  1.489031569107551, 1.4752480003481816, 1.495100842868565, 1.4766447499399793, 1.436179012670158,
                  1.4246868167075366, 1.3687113473195005, 1.279284813908499, 1.2831634988098926, 1.2221323778021431,
                  1.1880443307194573, 1.1772563765991473, 1.1249692646787497, 1.0818235675047765, 1.0629183271554359,
                  1.0990353430844322, 1.089687818274307, 1.1484582057182955, 1.2368813710604787, 1.3126982337148183,
                  1.337768956166502, 1.3377925993697275, 1.3639884643708715, 1.3630230253517754
                )

    print(costo,"kg")
