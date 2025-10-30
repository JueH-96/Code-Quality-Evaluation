import sys
from collections import defaultdict

MOD = 998244353

def main():
    n, m = map(int, sys.stdin.readline().split())
    operations = [tuple(map(int, sys.stdin.readline().split())) for _ in range(m)]
    
    dp = defaultdict(int)
    dp[(0, 0, 0, 0)] = 1  # initial state
    
    for P, V in operations:
        new_dp = defaultdict(int)
        for (l, r, a, b), cnt in dp.items():
            # Check prefix operation
            if V >= a:
                new_l = max(l, P)
                new_r = r
                new_a = max(a, V)
                new_b = b
                new_state = (new_l, new_r, new_a, new_b)
                new_dp[new_state] = (new_dp[new_state] + cnt) % MOD
            # Check suffix operation
            if V >= b:
                new_l = l
                new_r = min(r, P)
                new_a = a
                new_b = max(b, V)
                new_state = (new_l, new_r, new_a, new_b)
                new_dp[new_state] = (new_dp[new_state] + cnt) % MOD
        dp = new_dp
    
    total = sum(dp.values()) % MOD
    print(total)

if __name__ == "__main__":
    main()