def parse_input(file):
    with open(file, 'r') as f:
        lines = [line.rstrip() for line in f]

    a = [[0,0,0] for _ in range(3)]
    b = [0,0,0]
    for i in range(3):
        lines[i] = lines[i].replace(" ", "")

        ix = lines[i].find("x")
        if ix != -1:
            aux = lines[i][:ix]
            if len(aux) > 0:
                if aux[0] == "-":
                    if len(aux) == 1:
                        a[i][0] = -1
                    else:
                        a[i][0] = -int(aux[1:])
                elif aux[0] == "+":
                    if len(aux) == 1:
                        a[i][0] = 1
                    else:
                        a[i][0] = int(aux[1:])
                else:
                    a[i][0] = int(aux)
            else:
                a[i][0] = 1

        iy = lines[i].find("y")
        if iy != -1:
            aux = lines[i][ix+1:iy]
            if len(aux) > 0:
                if aux[0] == "-":
                    if len(aux) == 1:
                        a[i][1] = -1
                    else:
                        a[i][1] = -int(aux[1:])
                elif aux[0] == "+":
                    if len(aux) == 1:
                        a[i][1] = 1
                    else:
                        a[i][1] = int(aux[1:])
                else:
                    a[i][1] = int(aux)
            else:
                a[i][1] = 1

        iz = lines[i].find("z")
        if iz != -1:
            aux = lines[i][iy+1:iz]
            if len(aux) > 0:
                if aux[0] == "-":
                    if len(aux) == 1:
                        a[i][2] = -1
                    else:
                        a[i][2] = -int(aux[1:])
                elif aux[0] == "+":
                    if len(aux) == 1:
                        a[i][2] = 1
                    else:
                        a[i][2] = int(aux[1:])
                else:
                    a[i][2] = int(aux)
            else:
                a[i][2] = 1

        ib = lines[i].find("=")
        aux = lines[i][ib+1:]
        if aux[0] == "-":
            b[i] = -int(aux[1:])
        elif aux[0] == "+":
            b[i] = int(aux[1:])
        else:
            b[i] = int(aux)

    return a, b