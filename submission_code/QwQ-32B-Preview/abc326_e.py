def main():
    import sys
    MOD = 998244353

    # Read input
    input = sys.stdin.read().split()
    N = int(input[0])
    A = list(map(int, input[1:N+1]))
    A.append(0)  # A[N] = 0 for convenience

    # Precompute suffix sums S_A[x] = sum(A[x+1] to A[N])
    S_A = [0] * (N + 1)
    for x in range(N-1, -1, -1):
        S_A[x] = (S_A[x+1] + A[x+1]) % MOD

    # Compute modular inverses
    N_inv = pow(N, MOD-2, MOD)
    N1_inv = pow(N-1, MOD-2, MOD)

    # Initialize S_E[N] = 0
    S_E = [0] * (N + 1)

    # Compute S_E[x] and E[x] from x = N-1 down to x = 0
    E = [0] * (N + 1)
    for x in range(N-1, -1, -1):
        S_E[x] = (N * S_E[x+1] + S_A[x]) * N1_inv % MOD
        E[x] = (S_A[x] + S_E[x]) * N_inv % MOD

    # Output E[0]
    print(E[0])

if __name__ == "__main__":
    main()