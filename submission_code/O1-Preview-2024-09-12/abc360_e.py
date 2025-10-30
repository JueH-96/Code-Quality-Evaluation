# YOUR CODE HERE
MOD = 998244353

def modinv(a):
    return pow(a, MOD - 2, MOD)

def main():
    import sys
    import threading
    def solve():
        N, K = map(int, sys.stdin.readline().split())
        N_mod = N % MOD
        N_plus_1 = (N + 1) % MOD
        N_minus_1 = (N - 1) % MOD
        N_minus_2 = (N - 2) % MOD

        N_pow_K_mod = pow(N_mod, K, MOD)
        N_minus_2_pow_K_mod = pow(N_minus_2, K, MOD)

        P = (N_plus_1 * N_pow_K_mod - N_minus_1 * N_minus_2_pow_K_mod) % MOD
        Q = (2 * N_pow_K_mod) % MOD

        Q_inv = modinv(Q)
        R = (P * Q_inv) % MOD
        print(R)
    threading.Thread(target=solve).start()