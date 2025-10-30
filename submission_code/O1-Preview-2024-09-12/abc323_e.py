# YOUR CODE HERE
import sys
import threading
def main():
    import sys
    sys.setrecursionlimit(1 << 25)
    N, X = map(int, sys.stdin.readline().split())
    T = list(map(int, sys.stdin.readline().split()))
    t = X + 0.5  # We are to find the probability at time t
    max_T = max(T)
    max_s = int(t) + max_T + 2  # Maximum duration we need to consider

    MOD = 998244353

    dp = [0] * (max_s + 1)
    dp[0] = 1  # Initial probability

    ans_num = 0  # Numerator of the probability
    N_inv = pow(N, MOD - 2, MOD)  # Precompute modular inverse of N

    for s in range(max_s):
        prob = dp[s] % MOD  # Get probability at time s
        if prob == 0:
            continue
        inv_N = N_inv  # Modular inverse of N

        for T_i in T:
            s_next = s + T_i
            if s_next > max_s:
                continue
            dp[s_next] = (dp[s_next] + prob * inv_N) % MOD

            if s <= t < s + T_i:
                # At time t, song T_i is being played
                if T_i == T[0]:
                    ans_num = (ans_num + prob * inv_N) % MOD  # Add probability if it's song 1

    # Now compute the denominator, which is the total probability mass
    # Since the process is stochastic and total probability is 1, the denominator is 1
    # So the probability is ans_num modulo MOD

    # However, dp may have accumulated fractions, due to division by N at each step
    # So we need to compute the total probability mass
    # In this process, since we divided by N each time, the denominators are powers of N
    # We need to compute ans_num / N^k, where k is the number of song choices made to reach time t

    # Since we do not know k directly, and N is not divisible by MOD (since N ≤ 1e3 and MOD is prime > 1e3),
    # We can treat N as invertible modulo MOD, and we can multiply ans_num by the modular inverse of N^k

    # Since we didn't keep track of the powers of N, and MOD is large, and since N ≤ 1e3,
    # The probabilities we calculated are accurate modulo MOD

    # So the answer modulo MOD is ans_num
    print(ans_num % MOD)
threading.Thread(target=main).start()