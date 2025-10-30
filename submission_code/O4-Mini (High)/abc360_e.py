import sys
import threading

def main():
    data = sys.stdin.read().split()
    N = int(data[0])
    K = int(data[1])
    MOD = 998244353
    
    # Precompute inverses
    inv2 = (MOD + 1) // 2
    invN = pow(N, MOD - 2, MOD)
    
    # a = (N-2)/N  mod MOD
    a = (N - 2) * invN % MOD
    # a^K mod MOD
    ak = pow(a, K, MOD)
    
    # Expected value:
    # E = (N+1)/2 + (1-N)/2 * a^K
    #    = ((N+1) + (1-N)*a^K) * inv2  mod MOD
    res = ((N + 1) + (1 - N) * ak) % MOD
    res = res * inv2 % MOD
    
    print(res)

if __name__ == "__main__":
    main()