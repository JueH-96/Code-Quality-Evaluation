MOD = 998244353

def main():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    A = list(map(int, input[1:N+1]))
    A = [0] + A  # Make A 1-based

    if N == 0:
        print(0)
        return

    inv_N = pow(N, MOD-2, MOD)
    factor = ((N + 1) * inv_N) % MOD

    # Initialize C array
    C = [0] * (N + 2)  # C[0..N+1], but we use up to C[N]
    C[N] = A[N] % MOD

    for x in range(N-1, 0, -1):
        C[x] = (A[x] + factor * C[x + 1]) % MOD

    E = (C[1] * inv_N) % MOD
    print(E)

if __name__ == '__main__':
    main()