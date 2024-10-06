import operations as op

def cramer(a, b):
    if op.det(a) == 0:
        return "Cannot solve using Cramer"
    ca = op.copy_matrix(a)
    x = op.det(op.get_ak(ca, b, 'x')) / op.det(a)
    ca = op.copy_matrix(a)
    y = op.det(op.get_ak(ca, b, 'y')) / op.det(a)
    ca = op.copy_matrix(a)
    z = op.det(op.get_ak(ca, b, 'z')) / op.det(a)
    return x, y, z