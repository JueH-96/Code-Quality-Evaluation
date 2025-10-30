def main():
    import sys
    sys.setrecursionlimit(10**7)
    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    A = list(map(int, input_data[1:]))

    MOD = 998244353

    # Precompute modular inverse of N
    # Using Fermat's little theorem: invN = N^(MOD-2) mod MOD
    invN = pow(N, MOD-2, MOD)

    # S_A[i] = sum of A[i] + A[i+1] + ... + A[N]
    # We'll store S_A for i in [0..N], so S_A[N] = A_N, S_A[N+1] = 0
    # but we'll do 1-based indexing for A
    S_A = [0] * (N+2)
    for i in range(N, 0, -1):
        S_A[i] = (S_A[i+1] + A[i-1]) % MOD

    # E[k] = expected future payment if x=k
    # S_E[k] = E[k] + E[k+1] + ... + E[N]
    E = [0] * (N+1)
    S_E = [0] * (N+1)

    # Base case
    E[N] = 0
    S_E[N] = 0

    # Fill E[] and S_E[] in descending order
    for k in range(N-1, -1, -1):
        # E(k) = (S_A[k+1] + S_E[k+1]) / N  (all mod)
        E[k] = (S_A[k+1] + S_E[k+1]) % MOD
        E[k] = (E[k] * invN) % MOD
        S_E[k] = (S_E[k+1] + E[k]) % MOD

    # E(0) is the answer
    print(E[0])

# Call main() to run the solution
if __name__ == "__main__":
    main()