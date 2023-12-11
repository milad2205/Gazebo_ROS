import math

def get_x(x0, dx0, zc, t):
    g = 9.8
    Tc = math.sqrt(zc/g)
    tau = t/Tc
    x = x0*math.cosh(tau) + Tc*dx0*math.sinh(tau)
    return x

def get_dx(x0, dx0, zc, t):
    g = 9.8
    Tc = math.sqrt(zc / g)
    tau = t/Tc
    dx = (x0/Tc)*math.sinh(tau) + dx0*math.cosh(tau)
    return dx


def get_ddx(x0, dx0, zc, t):
    g = 9.8
    ddx = (g / zc) * get_x(x0, dx0, zc, t)
    return ddx


def get_orbital_energy(x, dx, zc):
    g = 9.8
    e = 0.5*dx**2-0.5*(g/zc)*x**2
    return e


def get_st(x0, dx0, zc, aim_e, aim_step, alpha, beta, lamda):
    t = 0.5
    step = aim_step
    g = 9.8

    x = get_x(x0, dx0, zc, t)
    dx = get_dx(x0, dx0, zc, t)
    e = get_orbital_energy(step - x, dx, zc)

    dfds = 2*beta*(step - aim_step)+(2*alpha*9.8)/zc*(e - aim_e)*(x-step)
    dfdt = 2*alpha*(e - aim_e)*dx*g/zc*(x + 1)
    step_new = step - lamda*dfds
    t_new = t - lamda*dfdt

    step_e = step_new - step
    t_e = t_new - t
    step = step_new
    t = t_new
    i = 1

    # f = alpha*(e-aim_e)**2 + beta*(step - aim_step)**2
    e_error = e - aim_e
    step_error = 0

    while abs(e_error) > 0.000001 or abs(t_e) > 0.000001:
        x = get_x(x0, dx0, zc, t)
        dx = get_dx(x0, dx0, zc, t)
        e = get_orbital_energy(step - x, dx, zc)

        dfds = 2*beta*(step - aim_step)+(2*alpha*9.8)/zc*(e - aim_e)*(x - step)
        dfdt = 2*alpha*(e - aim_e)*dx*g/zc*(x + 1)

        step_new = step - lamda*dfds
        t_new = t - lamda*dfdt

        step_e = step_new - step
        t_e = t_new - t

        step = step_new
        t = t_new
        e_error = e - aim_e
        step_error = step - aim_step
        i = i + 1

        if i > 1e7:
            break
    print("times",e_error,step_error)
    return  step, t


def eight_state(x, dx, E):
    if x > 0 and dx > 0 and E > 0:
        print('move forward:8')
        print('x=',x ,'dx=',dx ,'E=',E)
        return 8
    if x > 0 and dx > 0 and E < 0:
        print('move forward:7')
        print('x=',x ,'dx=',dx ,'E=',E)
        return 7
    if x > 0 and dx < 0 and E < 0:
        print('move forward:6')
        print('x=',x ,'dx=',dx ,'E=',E)
        return 6
    if x > 0 and dx < 0 and E > 0:
        print('move back:5')
        print('x=',x ,'dx=',dx ,'E=',E)
        return 5
    if x < 0 and dx < 0 and E > 0:
        print('move back :4')
        print('x=',x ,'dx=',dx ,'E=',E)
        return 4
    if x < 0 and dx < 0 and E < 0:
        print('move back :3')
        print('x=',x ,'dx=',dx ,'E=',E)
        return 3
    if x < 0 and dx > 0 and E < 0:
        print('move back :2')
        print('x=',x ,'dx=',dx ,'E=',E)
        return 2
    if x < 0 and dx > 0 and E > 0:
        print('move forward :1')
        print('x=',x ,'dx=',dx ,'E=',E)
        return 1


#achieve a e close aim_t aim_step
def get_st_with_e(x0, dx0, zc, aim_e, aim_step, aim_t, alpha, beta, gamma, lamda):
    # template gait
    t = aim_t
    step = aim_step
    g = 9.8
    # x0 = 0.1
    # dx0 = 0.1
    x = get_x(x0, dx0, zc, t)
    dx = get_dx(x0, dx0, zc, t)
    e = get_orbital_energy(step - x, dx, zc)
    f = alpha*(e-aim_e)**2 + beta*(step - aim_step)**2 + gamma*(t - aim_t)**2

    dfds = 2*beta*(step - aim_step)+(2*alpha*9.8)/zc*(e - aim_e)*(x-step)
    dfdt = 2*alpha*(e - aim_e)*dx*g/zc*(x + 1) + 2*gamma*(t-aim_t)

    step_new = step - lamda*dfds
    t_new = t - lamda*dfdt

    step_e = step_new - step
    t_e = t_new - t


    step = step_new
    t = t_new
    i = 1

    # f for optimal
    # f = alpha*(e-aim_e)**2 + beta*(step - aim_step)**2 + gamma*(t - aim_t)**2
    e_error = e - aim_e
    step_error = 0

    while abs(step_e) > 0.0001 or abs(t_e) > 0.0001:
        x = get_x(x0, dx0, zc, t)
        dx = get_dx(x0, dx0, zc, t)
        e = get_orbital_energy(step - x, dx, zc)

        dfds = 2*beta*(step - aim_step)+(2*alpha*9.8)/zc*(e - aim_e)*(x-step)
        dfdt = 2*alpha*(e - aim_e)*dx*g/zc*(x + 1) + 2*gamma*(t-aim_t)

        step_new = step - lamda*dfds
        t_new = t - lamda*dfdt

        step_e = step_new - step
        t_e = t_new - t

        step = step_new
        t = t_new
        e_error = e - aim_e
        step_error = step - aim_step
        t_error = t - aim_t
        i = i + 1
        if i > 1e7:
            break
    # print("times", t_error, step_error, i)
    return  step, t
