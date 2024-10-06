import operations as op

def get_cofactor(a, i, j):
    res = (-1) ** (i + j)
    minor = [[0, 0], [0, 0]]
    for row in range(3):
        for col in range(3):
            if row != i and col != j:
                minor[row - (row > i)][col - (col > j)] = a[row][col]
    res *= (minor[0][0] * minor[1][1] - minor[0][1] * minor[1][0])
    return res

def get_cofactor_matrix(a):
    cofactor_matrix = [[0,0,0],[0,0,0],[0,0,0]]
    for i in range(3):
        for j in range(3):
            cofactor_matrix[i][j] = get_cofactor(a, i, j)
    return cofactor_matrix

def adjugate(a):
    cofactor_matrix = get_cofactor_matrix(a)
    res = op.transpose(cofactor_matrix)
    return res

def inverse(a):
    det = op.det(a)
    if det == 0:
        return "Cannot invert matrix"
    adj = adjugate(a)
    for i in range(3):
        for j in range(3):
            adj[i][j] /= det
    return adj

def inverse_solution(a, b):
    inv = inverse(a)
    if type(inv) == str:
        return "Cannot solve using inverse"
    return op.m_v_mult(inv, b)