import sys
from collections import defaultdict

def main():
    MOD = 998244353
    N, K = map(int, sys.stdin.readline().split())
    S = sys.stdin.readline().strip()
    
    current_dp = defaultdict(int)
    current_dp[tuple()] = 1  # Initial state: empty tuple
    
    for c in S:
        next_dp = defaultdict(int)
        allowed = ['A', 'B'] if c == '?' else [c]
        for state, cnt in current_dp.items():
            for ch in allowed:
                if len(state) < K-1:
                    new_state = state + (ch,)
                    next_dp[new_state] = (next_dp[new_state] + cnt) % MOD
                else:
                    window = state + (ch,)
                    if window == window[::-1]:
                        continue
                    new_state = state[1:] + (ch,)
                    next_dp[new_state] = (next_dp[new_state] + cnt) % MOD
        current_dp = next_dp
        if not current_dp:
            break  # No possible states left
    
    print(sum(current_dp.values()) % MOD)

if __name__ == "__main__":
    main()