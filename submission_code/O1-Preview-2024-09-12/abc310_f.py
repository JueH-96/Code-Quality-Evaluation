# YOUR CODE HERE
import sys
import threading
def main():
    import sys
    sys.setrecursionlimit(1 << 25)
    N_and_rest = sys.stdin.read().split()
    N = int(N_and_rest[0])
    A_list = list(map(int, N_and_rest[1:N+1]))
    mod = 998244353

    dp = [0] * 11  # dp[s], s = 0..10
    dp[0] = 1  # Initially, sum 0 can be formed in 1 way

    for A_i in A_list:
        dp_new = [0] * 11
        max_v = min(A_i, 10)
        # Die i can have (A_i - max_v) ways to have value >10 or not contribute to sums up to 10
        for s in range(11):
            # Not including sum contributions from die i with value ≤10
            dp_new[s] = (dp_new[s] + dp[s] * (A_i - max_v)) % mod

        # Including die i's values from 1 to max_v
        for v in range(1, max_v +1):
            for s in range(11):
                if s + v <= 10:
                    dp_new[s + v] = (dp_new[s + v] + dp[s]) % mod
        dp = dp_new

    total_combinations = 1
    for A_i in A_list:
        total_combinations = (total_combinations * A_i) % mod

    y = dp[10] % mod
    x = total_combinations % mod

    # Since x is not divisible by mod, compute inverse of x modulo mod
    inv_x = pow(x, mod - 2, mod)

    z = (y * inv_x) % mod
    print(z)