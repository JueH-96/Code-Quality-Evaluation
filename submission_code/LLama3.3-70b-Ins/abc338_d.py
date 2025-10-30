import sys

def read_ints():
    return list(map(int, sys.stdin.readline().split()))

def main():
    N, M = read_ints()
    X = read_ints()

    # Calculate the total length of the tour without closing any bridge
    total_length = 0
    for i in range(M - 1):
        total_length += min(abs(X[i] - X[i + 1]), N - abs(X[i] - X[i + 1]))

    # Calculate the minimum length of the tour when a bridge is closed
    min_length = float('inf')
    for i in range(N):
        # Calculate the length of the tour when the i-th bridge is closed
        length = 0
        for j in range(M - 1):
            if (X[j] - X[j + 1]) % N == i + 1 or (X[j + 1] - X[j]) % N == N - i - 1:
                length += N
            else:
                length += min(abs(X[j] - X[j + 1]), N - abs(X[j] - X[j + 1]))
        min_length = min(min_length, length)

    print(min_length)

if __name__ == "__main__":
    main()