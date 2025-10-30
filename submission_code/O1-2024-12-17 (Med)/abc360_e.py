def main():
    import sys
    MOD = 998244353
    
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    K = int(data[1])
    
    # If there is only one ball (which is black), it never moves.
    # Hence the expected position is always 1.
    if N == 1:
        print(1)
        return
    
    # Modular inverse of 2 and of N
    inv2 = pow(2, MOD - 2, MOD)      # 1/2 mod
    invN = pow(N, MOD - 2, MOD)      # 1/N mod
    
    # r = 1 - 2*(1/N) = 1 - 2/N (mod)
    r = (1 - 2 * invN) % MOD
    
    # Compute r^K using fast exponentiation mod
    rK = pow(r, K, MOD)
    
    # We use the derived formula for E[X_K]:
    # E[X_K] = 1/2 * ( (N+1) + (1-N)*r^K ) in real arithmetic;
    # All operations are taken modulo MOD.
    Nplus1 = (N + 1) % MOD
    oneMinusN = (1 - N) % MOD
    
    val = (Nplus1 + oneMinusN * rK) % MOD
    ans = (inv2 * val) % MOD
    
    print(ans)

# Do not forget to call main!
if __name__ == "__main__":
    main()