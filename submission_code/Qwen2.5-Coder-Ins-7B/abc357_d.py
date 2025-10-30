# YOUR CODE HERE
import sys

MOD = 998244353

def main():
    N = int(sys.stdin.read().strip())
    
    # Calculate V_N % MOD
    V_N_mod = (N * pow(10, N, MOD - 1)) % MOD
    
    print(V_N_mod)

if __name__ == "__main__":
    main()