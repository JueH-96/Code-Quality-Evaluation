import sys

def read_ints():
    return list(map(int, sys.stdin.readline().split()))

def main():
    N, K = read_ints()
    X = read_ints()
    A = read_ints()

    # Adjust X to 0-indexed
    X = [x - 1 for x in X]

    # Perform the operation K times
    for _ in range(K):
        B = [A[X[i]] for i in range(N)]
        A = B

    print(' '.join(map(str, A)))

if __name__ == '__main__':
    main()