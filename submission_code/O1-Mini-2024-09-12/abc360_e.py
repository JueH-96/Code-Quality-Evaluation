# YOUR CODE HERE
import sys

def main():
    MOD = 998244353
    N_str, K_str = sys.stdin.read().split()
    N = int(N_str)
    K = int(K_str)
    
    if N ==1:
        print(1)
        return
    
    inv2 = pow(2, MOD-2, MOD)
    term1 = (N +1) * inv2 % MOD
    term2_factor = (N -1) * inv2 % MOD
    invN = pow(N, MOD-2, MOD)
    c = (N -2) * invN % MOD
    c_k = pow(c, K, MOD)
    E = (term1 - term2_factor * c_k) % MOD
    print(E)

if __name__ == "__main__":
    main()