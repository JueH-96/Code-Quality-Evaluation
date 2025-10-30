import sys
from collections import defaultdict

MOD = 998244353

def main():
    N, K = map(int, sys.stdin.readline().split())
    S = sys.stdin.readline().strip()
    
    possible_chars = []
    for c in S:
        if c == '?':
            possible_chars.append(['A', 'B'])
        else:
            possible_chars.append([c])
    
    current_dp = defaultdict(int)
    current_dp[''] = 1  # initial state
    
    for i in range(N):
        next_dp = defaultdict(int)
        for state, cnt in current_dp.items():
            for c in possible_chars[i]:
                if len(state) < K-1:
                    new_state = state + c
                    next_dp[new_state] = (next_dp[new_state] + cnt) % MOD
                else:
                    substr = state + c
                    if substr == substr[::-1]:
                        continue
                    new_state = state[1:] + c
                    next_dp[new_state] = (next_dp[new_state] + cnt) % MOD
        current_dp = next_dp
        if not current_dp:
            break  # no possible ways
    
    print(sum(current_dp.values()) % MOD)

if __name__ == "__main__":
    main()