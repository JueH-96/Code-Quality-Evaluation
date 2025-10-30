import sys
import threading

def main():
    import sys
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    M = int(data[1])
    MOD = 998244353

    # The number of ways to color a cycle of N vertices with M colors
    # such that adjacent vertices have different colors is given by
    #   (M-1)^N + (-1)^N * (M-1)
    # This is the chromatic polynomial of the cycle evaluated at M.
    base = M - 1
    term = pow(base, N, MOD)
    if N % 2 == 0:
        term = (term + base) % MOD
    else:
        term = (term - base) % MOD

    print(term)

if __name__ == "__main__":
    main()