import sys

def main():
    input = sys.stdin.read().split()
    N = int(input[0])
    L = int(input[1])
    R = int(input[2])
    A = list(map(int, input[3:3+N]))
    X = []
    for a in A:
        if a < L:
            X.append(str(L))
        elif a > R:
            X.append(str(R))
        else:
            X.append(str(a))
    print(' '.join(X))

if __name__ == '__main__':
    main()