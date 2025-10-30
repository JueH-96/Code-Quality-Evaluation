import sys
input = sys.stdin.read

def main():
    data = input().split()
    N = int(data[0])
    S = list(data[1])
    Q = int(data[2])
    operations = data[3:]

    for i in range(Q):
        t = int(operations[3 * i])
        x = int(operations[3 * i + 1])
        c = operations[3 * i + 2]

        if t == 1:
            S[x - 1] = c
        elif t == 2:
            S = [char.lower() for char in S]
        elif t == 3:
            S = [char.upper() for char in S]

    print(''.join(S))

if __name__ == "__main__":
    main()