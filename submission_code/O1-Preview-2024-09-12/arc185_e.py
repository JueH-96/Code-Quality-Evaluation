# YOUR CODE HERE
import sys
import threading
import math
from collections import defaultdict

def main():
    import sys

    sys.setrecursionlimit(1 << 25)
    input = sys.stdin.readline
    N_and_rest = sys.stdin.read().split()
    N = int(N_and_rest[0])
    A = list(map(int, N_and_rest[1:N+1]))
    MOD = 998244353

    N = len(A)

    answer = []
    dp = dict()  # gcd -> cumulative count
    dp_score = dict()  # gcd -> cumulative score

    cumulative_answer = []
    cumulative_total = 0

    for idx, a in enumerate(A):
        new_dp = defaultdict(int)
        new_dp_score = defaultdict(int)
        # Start a new subsequence with only a
        new_dp[a] +=1
        new_dp_score[a] += 0  # Score is 0 for length 1 sequences

        # Extend previous subsequences
        for g, count in dp.items():
            g_new = math.gcd(g,a)
            c = count
            s = dp_score[g]
            incr = c * math.gcd(a, g)
            new_dp[g_new] += c
            new_dp_score[g_new] += (s + incr) % MOD

        dp = new_dp
        dp_score = new_dp_score

        total = sum(dp_score.values()) % MOD
        cumulative_total = (cumulative_total + total) % MOD
        cumulative_answer.append(cumulative_total)

    for ans in cumulative_answer:
        print(ans % MOD)


threading.Thread(target=main).start()