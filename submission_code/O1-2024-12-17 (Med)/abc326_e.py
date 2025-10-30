def main():
    import sys
    MOD = 998244353
    
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    A = list(map(int, data[1:]))

    # Compute the modular inverse of N
    # i.e. the number invN such that (N * invN) % MOD == 1
    invN = pow(N, MOD-2, MOD)

    # r = (N+1)/N in modular arithmetic
    r = (N + 1) % MOD
    r = (r * invN) % MOD

    # We'll compute E[x] in reverse order using:
    #   E[x] = r * E[x+1] + A[x]/N
    # for x = N-1 down to 0, with E[N] = 0
    E_next = 0  # This represents E[N], which is 0
    for i in range(N-1, -1, -1):
        E_i = (r * E_next + A[i] * invN) % MOD
        E_next = E_i

    # E_next after the loop ends is E[0], which is our answer
    print(E_next)

# Don't forget to call main()!
if __name__ == "__main__":
    main()