import sys
from collections import defaultdict

MOD = 998244353

def main():
    N, *rest = map(int, sys.stdin.read().split())
    A = rest[:N]
    
    dp = defaultdict(int)
    dp[0] = 1  # initial state: no dice processed, subset sums empty
    
    for a_i in A:
        new_dp = defaultdict(int)
        for state, count in dp.items():
            # Handle a_i > 10
            cnt_high = max(0, a_i - 10)
            new_dp[state] = (new_dp[state] + count * cnt_high) % MOD
            
            # Handle a_i <= 10
            max_a = min(a_i, 10)
            for a in range(1, max_a + 1):
                if a == 10:
                    continue
                # Check if (10 - a) is present in the current state
                if (state & (1 << (10 - a))) != 0:
                    continue
                # Compute new state
                shifted = (state << a) & 0x7ff  # Mask to 11 bits
                new_S = (state | (1 << a) | shifted) & 0x7ff
                # Ensure 10 is not in new_S
                if (new_S & (1 << 10)) == 0:
                    new_dp[new_S] = (new_dp[new_S] + count) % MOD
        dp = new_dp
    
    # Compute bad outcomes
    bad = sum(dp.values()) % MOD
    
    # Compute total outcomes modulo MOD
    total = 1
    for x in A:
        total = (total * x) % MOD
    
    # Compute favorable outcomes
    favorable = (total - bad) % MOD
    
    # Compute the answer as favorable * inverse(total) mod MOD
    inv_total = pow(total, MOD - 2, MOD)
    ans = (favorable * inv_total) % MOD
    print(ans)

if __name__ == '__main__':
    main()