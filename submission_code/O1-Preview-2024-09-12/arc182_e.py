# YOUR CODE HERE
import sys
import threading

def main():
    import math

    N, M, C, K = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))
    s0 = min(A)
    d = C % M

    if d == 0:
        s_k = s0 % M
        total = (s_k) * K
        print(total)
        return

    g = math.gcd(d, M)
    L = M // g  # Length of one full period
    T = K // L  # Number of full periods
    R = K % L   # Remaining steps

    # Function to compute sum_{k=0}^{n-1} floor( (a * k + b ) / m )
    def floor_sum(n, a, b, m):
        res = 0
        while True:
            if a >= m:
                res += (n - 1) * n * (a // m) // 2
                a %= m
            if b >= m:
                res += n * (b // m)
                b %= m
            y_max = (a * n + b) // m
            if y_max == 0:
                return res
            x_max = ((m - b - 1) // a) + 1
            if x_max < n:
                res += (n - x_max) * y_max
                n = x_max
            a, m = m, a
            b = (a - b % a) % a
        return res

    # Sum over one period
    L_s0 = s0
    L_n = L
    L_d = d
    L_m = M

    S_s = L_s0 * L_n + L_d * L_n * (L_n - 1) // 2
    W = floor_sum(L_n, L_d, L_s0, L_m)
    S_L = S_s - L_m * W

    # Sum over remaining steps
    R_s0 = s0
    R_n = R
    R_d = d
    R_m = M

    S_s_R = R_s0 * R_n + R_d * R_n * (R_n - 1) // 2
    W_R = floor_sum(R_n, R_d, R_s0, R_m)
    S_R = S_s_R - R_m * W_R

    total = T * S_L + S_R
    print(total)

threading.Thread(target=main).start()