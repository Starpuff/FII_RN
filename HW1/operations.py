def det(m):
    return m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) + m[1][0] * (m[2][1] * m[0][2] - m[0][1] * m[2][2]) + m[2][0] * (m[0][1] * m[1][2] - m[1][1] * m[0][2])

def trace(m):
    return m[0][0] + m[1][1] + m[2][2]

def norm(v):
    return (v[0] ** 2 + v[1] ** 2 + v[2] ** 2) ** 0.5

def transpose(m):
    return [[m[0][0], m[1][0], m[2][0]], [m[0][1], m[1][1], m[2][1]], [m[0][2], m[1][2], m[2][2]]]

def m_v_mult(m, v):
    return [m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2], m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2], m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2]]

def get_ak(m, v, k):
    if k == 'x':
        for i in range(3):
            m[i][0] = v[i]
        return m
    elif k == 'y':
        for i in range(3):
            m[i][1] = v[i]
        return m
    elif k == 'z':
        for i in range(3):
            m[i][2] = v[i]
        return m
    return ("Invalid key")


def copy_matrix(a):
    ca = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    for i in range(3):
        for j in range(3):
            ca[i][j] = a[i][j]
    return ca