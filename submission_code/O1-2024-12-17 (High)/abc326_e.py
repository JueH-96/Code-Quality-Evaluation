def main():
    import sys
    MOD = 998244353

    data = sys.stdin.read().strip().split()
    N = int(data[0])
    A = list(map(int, data[1:]))

    # Compute the modular inverse of N using Fermat's little theorem
    invN = pow(N, MOD - 2, MOD)

    # We'll maintain an array G where
    #   G[i] = Σ (A[k] + S(k)) for k from i to N, in the sense of the original derivation,
    # but effectively it can be computed via the recurrence:
    #   G[i] = G[i+1]*(1 + invN) + A[i-1]   (all operations mod MOD)
    #
    # In the end, the expected value from state x=0 is S(0) = (1/N) * G(1).
    # We need G of length at least N+2 so that G[N+1] can be safely referenced.

    G = [0] * (N + 2)
    factor = (1 + invN) % MOD  # This is (1 + 1/N) mod MOD

    # Build G[] in descending order
    for i in reversed(range(1, N + 1)):
        G[i] = (G[i + 1] * factor + A[i - 1]) % MOD

    # Finally, the answer is (1/N) * G(1) mod MOD
    ans = (invN * G[1]) % MOD
    print(ans)

# Don't forget to call main() 
if __name__ == "__main__":
    main()