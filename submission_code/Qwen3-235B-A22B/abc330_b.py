import sys

def main():
    input = sys.stdin.read().split()
    n = int(input[0])
    L = int(input[1])
    R = int(input[2])
    A = list(map(int, input[3:3+n]))
    result = []
    for a in A:
        if a < L:
            result.append(L)
        elif a > R:
            result.append(R)
        else:
            result.append(a)
    print(' '.join(map(str, result)))

if __name__ == "__main__":
    main()