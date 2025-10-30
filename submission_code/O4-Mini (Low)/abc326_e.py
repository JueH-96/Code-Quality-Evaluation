import sys
import threading
def main():
    import sys
    sys.setrecursionlimit(10**7)
    MOD = 998244353

    data = sys.stdin.read().split()
    N = int(data[0])
    A = [0] + list(map(int, data[1:]))

    invN = pow(N, MOD-2, MOD)
    # f[x] = expected remaining sum when current max = x
    # f[x] = (1/N) * sum_{y = x+1..N} (A[y] + f[y])
    f = [0] * (N+1)
    # We'll maintain S_A = sum_{y=x+1..N} A[y], S_f = sum_{y=x+1..N} f[y]
    S_A = 0
    S_f = 0

    # Start from x = N down to 0
    for x in range(N, -1, -1):
        if x < N:
            total = S_A + S_f
            if total >= MOD: total %= MOD
            f[x] = total * invN % MOD
        # Now include x in the sums for the next iteration
        # but only if x>=1 do we have A[x]
        if x >= 1:
            S_A += A[x]
            if S_A >= MOD: S_A -= MOD
            S_f += f[x]
            if S_f >= MOD: S_f -= MOD

    # Answer is f[0]
    print(f[0] % MOD)

if __name__ == "__main__":
    main()