def parse_input(file):
    with open(file, 'r') as f:
        lines = [line.replace(" ", "").rstrip() for line in f]

    a = [[0, 0, 0] for _ in range(3)]
    b = [0, 0, 0]

    for i in range(len(lines)):
        c = 1
        j = 0

        while j < len(lines[i]):
            if lines[i][j] == '-':
                c = -1
                j += 1
            elif lines[i][j] == '+':
                c = 1
                j += 1
            elif lines[i][j].isdigit():
                first = j
                while j < len(lines[i]) and lines[i][j].isdigit():
                    j += 1
                c *= int(lines[i][first:j])
            elif lines[i][j] == 'x':
                a[i][0] = c
                c = 1
                j += 1
            elif lines[i][j] == 'y':
                a[i][1] = c
                c = 1
                j += 1
            elif lines[i][j] == 'z':
                a[i][2] = c
                c = 1
                j += 1
            elif lines[i][j] == '=':
                j += 1
                if lines[i][j] == '-':
                    c = -1
                    j += 1
                else:
                    c = 1

                first = j
                while j < len(lines[i]) and lines[i][j].isdigit():
                    j += 1
                b[i] = c * int(lines[i][first:j])

            else:
                j += 1

    return a, b
