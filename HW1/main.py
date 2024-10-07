import parser
import cramer
import inversion


if __name__ == '__main__':
    a, b = parser.parse_input("input.txt")
    print(a)
    print(b)
    print(cramer.cramer(a, b))
    print(inversion.inverse_solution(a, b))
