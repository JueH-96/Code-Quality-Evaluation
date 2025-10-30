# YOUR CODE HERE
import sys

def main():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    S = list(input[1])
    Q = int(input[2])
    operations = input[3:]

    case = 0  # 0: normal, 1: all lowercase, 2: all uppercase

    for i in range(Q):
        t = int(operations[i * 3])
        x = int(operations[i * 3 + 1])
        c = operations[i * 3 + 2]

        if t == 1:
            if case == 0:
                S[x - 1] = c
            elif case == 1:
                S[x - 1] = c.lower()
            elif case == 2:
                S[x - 1] = c.upper()
        elif t == 2:
            if case == 2:
                case = 0
            elif case == 0:
                case = 1
        elif t == 3:
            if case == 1:
                case = 0
            elif case == 0:
                case = 2

    if case == 1:
        S = [char.lower() for char in S]
    elif case == 2:
        S = [char.upper() for char in S]

    print(''.join(S))

if __name__ == "__main__":
    main()