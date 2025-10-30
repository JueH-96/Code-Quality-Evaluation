import sys
import threading

def main():
    import sys
    data = sys.stdin.read().split()
    it = iter(data)
    n = int(next(it))
    A = [int(next(it)) for _ in range(n)]
    S = [0] * (n - 1)
    T = [0] * (n - 1)
    for i in range(n - 1):
        S[i] = int(next(it))
        T[i] = int(next(it))
    # Greedily convert as much as possible at each stage
    for i in range(n - 1):
        # number of times we can convert currency i
        k = A[i] // S[i]
        # consume
        A[i] -= k * S[i]
        # produce in i+1
        A[i+1] += k * T[i]
    # Answer is final currency N
    print(A[n-1])

if __name__ == "__main__":
    main()