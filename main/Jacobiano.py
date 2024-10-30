import numpy as np
import sympy

# Define symbolic variables.  Incluyendo constantes
i = sympy.Symbol('i', integer=True)
N = 53  # Define N como un entero (valor de prueba)
Zp_I = sympy.Symbol('Zp_I')
Zp_F = sympy.Symbol('Zp_F')
v = sympy.Function('v')
gamma = sympy.Function('gamma')
m = sympy.Function('m')
t = sympy.Function('t')
s = sympy.Function('s')
F_N_MCL = sympy.Function('F_N_MCL')
rho = sympy.Function('rho')
Cx_0 = sympy.Symbol('Cx_0')
k = sympy.Symbol('k')
Cz = sympy.Function('Cz')
S_REF = sympy.Symbol('S_REF')
g_0 = sympy.Symbol('g_0')
lambda_ = sympy.Function('lambda_') # evitar conflicto con palabra clave lambda
eta = sympy.Symbol('eta')
R = sympy.Symbol('R')
L_z = sympy.Symbol('L_z') # Mantener consistencia con el código anterior
Ts_0 = sympy.Symbol('Ts_0')
rho_0 = sympy.Symbol('rho_0')
alpha_0 = sympy.Symbol('alpha_0')
CI = sympy.Symbol('CI')
M_CRZ = sympy.Symbol('M_CRZ')
s_F = sympy.Symbol('s_F')
CAS_I = sympy.Symbol('CAS_I')
m_I = sympy.Symbol('m_I')

Zp = sympy.Lambda(i, Zp_I + i * (Zp_F - Zp_I) / (N - 1))

# Define rho(Zp(i))
rho = sympy.Lambda(i, rho_0*((Ts_0 + L_z*Zp(i))/Ts_0)**(alpha_0 - 1))

# Define F_N_MCL(Zp(i))
F_N_MCL = sympy.Lambda(i, 140000 - 2.53*Zp(i))



# Define las ecuaciones usando Zp(i) directamente
g_v = (v(i+1) - v(i))/(Zp(i+1) - Zp(i)) - \
    (1/2) * ( (lambda_(i+1) * F_N_MCL(i+1)) / (m(i+1) * v(i+1) * sympy.sin(gamma(i+1))) - \
            ( (1/2) * rho(i+1) * v(i+1) * S_REF * (Cx_0 + k*Cz(i+1)**2) ) / (m(i+1) * sympy.sin(gamma(i+1))) - \
             g_0 / v(i+1) + \
            (lambda_(i) * F_N_MCL(i)) / (m(i) * v(i) * sympy.sin(gamma(i))) - \
            ( (1/2) * rho(i) * v(i) * S_REF * (Cx_0 + k*Cz(i)**2) ) / (m(i) * sympy.sin(gamma(i))) - \
             g_0 / v(i) )

g_gamma = (gamma(i+1) - gamma(i))/(Zp(i+1) - Zp(i)) - \
    (1/2) * ( ( (1/2)*rho(i+1)*S_REF*Cz(i+1) ) / (m(i+1) * sympy.sin(gamma(i+1))) - \
              g_0 / (v(i+1)**2 * sympy.tan(gamma(i+1))) + \
              ( (1/2)*rho(i)*S_REF*Cz(i) ) / (m(i) * sympy.sin(gamma(i))) - \
              g_0 / (v(i)**2 * sympy.tan(gamma(i))) )

g_m = (m(i+1) - m(i))/(Zp(i+1) - Zp(i)) + \
      (1/2) * eta * ( (lambda_(i+1) * F_N_MCL(i+1)) / (v(i+1) * sympy.sin(gamma(i+1)))  + \
                    (lambda_(i) * F_N_MCL(i)) / (v(i) * sympy.sin(gamma(i))) )

g_t = (t(i+1) - t(i))/(Zp(i+1) - Zp(i)) - \
      (1/2) * ( 1/(v(i+1)*sympy.sin(gamma(i+1))) + 1/(v(i)*sympy.sin(gamma(i))) )

g_s = (s(i+1) - s(i))/(Zp(i+1) - Zp(i)) - \
      (1/2) * ( 1/sympy.tan(gamma(i+1)) + 1/sympy.tan(gamma(i)) )


def derivada_parcial(funcion, variable, i_funcion, i_variable):
    """
    Calcula la derivada parcial de una función respecto a una variable con índices específicos.
    """
    derivada = funcion.subs(i, i_funcion).diff(variable(i_variable)).simplify().trigsimp()
    return derivada


def calcular_jacobiano(N):
    """Calcula el Jacobiano de las restricciones (versión corregida)."""

    funciones = [g_v, g_gamma, g_m, g_t, g_s]
    variables = [v, gamma, m, t, s, Cz, lambda_]

    num_funciones = len(funciones) * (N - 1)
    num_variables = len(variables) * (N - 1)

    jacobiano = sympy.zeros(num_funciones, num_variables)

    for i_funcion in range(N - 1):
        for j_variable in range(N - 1):  # j_variable va de 0 a N-2, representando índices de 1 a N-1
            for k_func, func in enumerate(funciones):
                for l_var, var in enumerate(variables):
                    #  Se suma 1 a j_variable al derivar para representar índices de 1 a N-1
                    derivada = derivada_parcial(func, var, i_funcion, j_variable + 1)  # <--- CORRECCIÓN
                    jacobiano[i_funcion * len(funciones) + k_func, j_variable * len(variables) + l_var] = derivada

    #return jacobiano
    return jacobiano, num_variables



# Valores numéricos (con unidades correctas y valores para N=5)

valores_numericos = {
    Zp_I: 10000 * 0.3048,  # Convertir ft a m
    Zp_F: 36000 * 0.3048,  # Convertir ft a m
    Cx_0: 0.014,
    k: 0.09,
    S_REF: 120,
    g_0: 9.80665,
    eta: 0.06 / 3600,  # kg/(N*s)
    R: 287.05287,
    L_z: -0.0065,
    Ts_0: 288.15,
    rho_0: 1.225,
    alpha_0: -9.80665 / (287.05287 * -0.0065),  # Valor calculado, corregir signo
    m_I: 60000,  # kg
    CAS_I: 250 * 0.514444,  # Convertir kt a m/s
    CI: 30 / 60, # kg/s
    M_CRZ : 0.8,
    s_F : 400000
}

def generar_valores_numericos(N):
    """Genera un diccionario con valores numéricos de ejemplo para un N dado."""

    valores_numericos = {
        Zp_I: 10000 * 0.3048,  # Convertir ft a m
        Zp_F: 36000 * 0.3048,  # Convertir ft a m
        Cx_0: 0.014,
        k: 0.09,
        S_REF: 120,
        g_0: 9.80665,
        eta: 0.06 / 3600,  # kg/(N*s)
        R: 287.05287,
        L_z: -0.0065,
        Ts_0: 288.15,
        rho_0: 1.225,
        alpha_0: -9.80665 / (287.05287 * -0.0065),
        m_I: 60000,  # kg
        CAS_I: 250 * 0.514444,  # Convertir kt a m/s
        CI: 30 / 60,
        M_CRZ: 0.8,
        s_F: 400000
    }

    # Generar valores para las variables
    for i in range(1, N):
        velo = [136.72340757119528, 139.57664297548615, 151.95327777594713, 137.59477992202238, 144.6064094307896, 157.97701229735156, 165.27849090941032, 176.29415090821374, 186.22651828660727, 201.30525703185089, 215.95044899534938, 223.48183794549536, 210.68440649065496, 183.27145147267578, 191.76996308400928, 215.08825582934148, 207.09020190557808, 205.43367394272582, 206.13765224810723, 201.98441712711158, 198.12722666850672, 196.56099273807874, 201.18684708393573, 203.90763941834703, 208.15028179869384, 209.06351348914382, 209.3964542398741, 211.25542058078005, 204.54249749853912, 200.98452348169766, 205.7363863439262, 206.36957013412018, 206.32557016036867, 208.27815402907336, 209.56145513496713, 206.63319895040152, 205.33608639320803, 207.56256709801377, 208.41366157979397, 208.5413906727497, 205.28769469147326, 203.740890192529, 209.38156245307098, 210.122389671683, 205.2721070200644, 206.08194278333343, 209.63072230617183, 209.82289927390238, 208.79120219889498, 208.96165388285027, 208.70713448253298, 209.5264379536008]
        valores_numericos[v(i)] = velo[i-1]
        gammas = [2.357475185194937, 1.4639181456914254, 1.9835893762142731, 2.0493370893489584, 1.38241848944349, 2.2821959679097055, 1.7985805894129685, 2.0111785885536575, 1.623401310197683, 1.7538874956851358, 1.5585827017897296, 1.4021457202904453, 1.3899112381927174, 1.2681958896569596, 0.7544763724918945, 1.1647747412107117, 1.5575459896346395, 1.662019265769711, 2.474555301921813, 2.535867596633972, 2.68445533298746, 2.0299961230360375, 2.0790208375025867, 1.6585148384929018, 1.6063681510651249, 2.0666266852569715, 2.160879375242484, 2.1283352831552778, 1.4656326842060838, 1.709762423000586, 1.870584031925758, 2.2082967858355524, 2.2479942572093954, 2.0305464206368815, 2.395759657920259, 2.273092289158334, 1.6503189935778988, 1.885643032416091, 2.038571643986672, 1.7811749557283811, 1.604021199033904, 1.3203166102074078, 0.8829771186669655, 1.3135494660474156, 1.6275240837885119, 0.8636675185318708, 0.9521778200440366, 1.206811016215501, 1.0642462559012336, 0.8301015305970634, 0.8665345507319031, 0.4201279875872637]
        valores_numericos[gamma(i)] = gammas[i-1]  # Ejemplo: gamma(i) disminuye linealmente
        eme = [59987.327597390504, 59948.48693655946, 59900.18468828229, 59882.68009313893, 59838.13945437962, 59790.75136012878, 59756.06864815279, 59718.06473036222, 59681.906426976544, 59640.841563076865, 59601.20466131766, 59567.403185181225, 59552.59648935435, 59548.98947263323, 59499.64718939284, 59436.572442162855, 59415.9327765761, 59392.34013853172, 59369.73204873689, 59354.96592942957, 59340.093039185, 59321.425790252346, 59295.06370479483, 59269.51903020116, 59240.742293521515, 59217.095347772825, 59196.45192090064, 59174.633774728834, 59158.07866158463, 59135.89028256646, 59107.733997169504, 59085.78461341217, 59065.896160038465, 59043.63014831752, 59022.43644192277, 59006.13336208692, 58985.448762830405, 58959.73091126392, 58937.223139161964, 58915.11501717073, 58894.17577114624, 58868.21825963003, 58828.25939081831, 58793.274615632916, 58770.86372320875, 58736.84176960487, 58693.855523935315, 58659.27086154628, 58627.42290572583, 58588.585405653874, 58546.73829184889, 58486.30559340387]
        valores_numericos[m(i)] = eme[i-1]  # Ejemplo: m(i) disminuye linealmente
        te = [19.878025407329535, 54.13782074064494, 89.31861633847704, 118.72742096069673, 155.35167765362814, 188.6658633903532, 214.96234555470858, 241.45840286341962, 267.7131097821052, 294.01829270066185, 318.8810550896373, 345.280881982267, 373.58174087195755, 406.64264290999296, 454.6808829533357, 501.38690018377946, 531.7679589123579, 557.5973272894895, 578.5449812574625, 595.3108064866283, 611.7333289408978, 630.527653969325, 651.5085081423822, 674.4199973361449, 699.9008150650589, 722.6300050575294, 742.0156066139847, 761.013908858853, 784.8335012456099, 811.5911440548738, 835.1909386041623, 855.7252367934414, 874.3647979110121, 893.7332890903624, 912.3984650814398, 930.0551908233995, 951.8199087296368, 975.4088713586025, 996.4396353379736, 1018.0578074282326, 1042.6020517861746, 1071.537681576293, 1110.6334125615454, 1149.3250245068007, 1177.6696775434032, 1214.560886705142, 1260.0897142334927, 1298.468664068169, 1334.6651193567693, 1378.6394933064084, 1427.0216239796478, 1499.3698309784822]
        valores_numericos[t(i)] = te[i-1]  # Ejemplo: t(i) aumenta linealmente
        ese = [2791.075339732528, 7532.519789494883, 12616.614746781695, 16864.579413895368, 22051.905325392294, 27025.85905368484, 31282.666392953983, 35792.52065686046, 40559.450474724305, 45638.940018023044, 50828.19373601134, 56630.281091912664, 62765.96788091624, 69224.38974745343, 78278.72173921666, 87633.00540441576, 94059.60929429611, 99385.73240989383, 103692.3003020368, 107110.35398133523, 110392.96446574421, 114096.73823586038, 118265.45889180453, 122906.97786242656, 128154.9406318216, 132892.6874432762, 136945.90362227216, 140938.99652324652, 145872.73931399218, 151299.384362168, 156093.1419885742, 160321.08589506982, 164164.40989068188, 168177.61416426534, 172073.2266759096, 175743.63704563578, 180221.99162745773, 185087.72191764205, 189458.94344288993, 193963.44673184512, 199037.39713394133, 204950.98455048355, 213045.6429420278, 221157.0184923477, 227048.74825323658, 234639.34999608074, 244096.9952295546, 252144.24480571522, 259717.71857095446, 268902.15446125774, 279005.02891911054, 294143.6452398516]
        valores_numericos[s(i)] = ese[i-1]  # Ejemplo: s(i) aumenta linealmente
        Czz = [0.5656800434996644, 0.581445287169397, 0.48430848038692215, 0.6150722838701199, 0.5491391973930075, 0.500980760642712, 0.415433526828226, 0.42399453499051437, 0.32938199854076855, 0.3434236641844473, 0.24351758600741394, 0.28732639154375006, 0.2701960022563664, 0.42412425808803456, 0.35373444041013663, 0.32859447149981985, 0.2915685172808895, 0.3790740001595318, 0.2925321948827734, 0.41371091462657716, 0.3284862052356354, 0.42748905400517756, 0.3495675248445142, 0.4016494187326902, 0.3470925956404801, 0.40885669922405854, 0.3406211657007417, 0.41402427209989995, 0.37695979633578824, 0.47304689036665215, 0.37444619748467267, 0.48173536723011096, 0.3705645804437795, 0.48167785597241036, 0.37676730953824344, 0.5032086992668854, 0.41997023985970583, 0.5214047441050225, 0.407995567489406, 0.526001996751008, 0.4521406288673375, 0.5540014819850765, 0.4696025543095781, 0.5521746582650408, 0.4805618204350393, 0.566908699685594, 0.5074824321132598, 0.5796239185391056, 0.5231315234566749, 0.592962291600977, 0.5521226137193213, 0.5963086325836461]
        valores_numericos[Cz(i)] = Czz[i-1]  # Ejemplo: Cz(i) disminuye linealmente
        lamdas = [0.0033767803667336116, 0.9883332416698697, 0.363105414472219, 0.2863733218723919, 0.9440556472058956, 0.5086463159797037, 0.9376654830071243, 0.6694960132075662, 0.9010201261895642, 0.9200739740838184, 0.9611540330431319, 0.580343638544006, 0.06694138875301003, 0.06683731434062479, 0.9985700829659806, 0.5819998678617202, 0.2369208911153714, 0.9673266434859726, 0.30763363525757415, 0.8651594515931619, 0.33869513546134844, 0.9358045008098441, 0.7972319241066789, 0.7707806676222754, 0.8371604516901925, 0.6377742173218477, 0.9329906097083915, 0.7768186402295985, 0.3569592773260758, 0.9627469312988324, 0.8951067720130466, 0.7906561971900374, 0.9334885097472201, 0.9531096228992094, 0.9398226290303711, 0.636592752806503, 0.9596768063885014, 0.9584939005936064, 0.9600821396555757, 0.9124313881561308, 0.6886300120284649, 0.9892026368070445, 0.9934791084186825, 0.73689438879606, 0.8710982733717808, 0.998161749652333, 0.9880451254779636, 0.943427918131619, 0.9925017633010199, 0.9993474349203401, 0.995132322436318, 0.9891117812140342]
        valores_numericos[lambda_(i)] = lamdas[i-1]  # Ejemplo: lambda_(i) disminuye linealmente



    return valores_numericos

valores_numericos = generar_valores_numericos(N)

# Calcular constantes con subíndice 0 (con unidades consistentes)
v0 = sympy.sqrt(7 * valores_numericos[R] * (valores_numericos[Ts_0] + valores_numericos[L_z] * valores_numericos[Zp_I]) *
              (((1 + valores_numericos[CAS_I]**2 / (7 * valores_numericos[R] * valores_numericos[Ts_0]))**(3.5) - 1) *
               (valores_numericos[Ts_0] / (valores_numericos[Ts_0] + valores_numericos[L_z] * valores_numericos[Zp_I]))**(-valores_numericos[alpha_0]) + 1)**(1/3.5) - 1)


rho0_val = rho(0).subs(valores_numericos).evalf()


Cz0 = (valores_numericos[m_I] * valores_numericos[g_0]) / (0.5 * rho0_val * v0**2 * valores_numericos[S_REF])


gamma0 = sympy.asin((F_N_MCL(0).subs(valores_numericos).evalf() - 0.5 * rho0_val * v0**2 * valores_numericos[S_REF] * (valores_numericos[Cx_0] + valores_numericos[k] * Cz0**2)) / (valores_numericos[m_I] * valores_numericos[g_0]))


valores_numericos_0 = {
    v(0): v0,
    gamma(0): gamma0,
    m(0): valores_numericos[m_I],
    t(0): 0,
    s(0): 0,
    lambda_(0): 1,
    Cz(0): Cz0
}




# Resto del código (similar al anterior)
jacobiano_N5, num_variables = calcular_jacobiano(N)


jacobiano_numerico = jacobiano_N5.subs(valores_numericos).subs(valores_numericos_0)



jacobiano_numpy = np.array([[float(sympy.re(expr).evalf()) if expr.is_complex else float(expr.evalf()) if expr.is_number else float(sympy.re(expr.subs({s: 0 for s in expr.free_symbols})).evalf()) for expr in row] for row in jacobiano_numerico.tolist()], dtype=float)


rango = np.linalg.matrix_rank(jacobiano_numpy)
print("Rango del Jacobiano:", rango)

variables_independientes = num_variables - rango
print("Número de variables independientes:", variables_independientes)

