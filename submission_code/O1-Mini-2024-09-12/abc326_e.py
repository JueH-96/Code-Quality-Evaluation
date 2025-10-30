# YOUR CODE HERE
import sys
import sys
def main():
    import sys
    sys.setrecursionlimit(1 << 25)
    MOD = 998244353
    input = sys.stdin.read
    data = input().split()
    N = int(data[0])
    A = list(map(int, data[1:N+1]))
    inv_N = pow(N, MOD-2, MOD)
    f = [0] * (N +2)  # f[N] =0
    for x in range(N-1, -1, -1):
        term1 = A[x]  # A[x+1] in 1-based, A[x] in 0-based
        term2 = (N +1) * f[x+1] % MOD
        f[x] = (term1 + term2) * inv_N % MOD
    print(f[0])
if __name__ == "__main__":
    main()