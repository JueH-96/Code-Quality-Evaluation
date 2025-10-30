import sys
from collections import defaultdict

def main():
    MOD = 998244353
    N, K = map(int, sys.stdin.readline().split())
    S = sys.stdin.readline().strip()
    
    prev_dp = defaultdict(int)
    prev_dp[''] = 1  # Initial state is empty string with count 1
    
    for c_char in S:
        curr_dp = defaultdict(int)
        for state_str, cnt in prev_dp.items():
            if c_char == '?':
                next_chars = ['A', 'B']
            else:
                next_chars = [c_char]
            for nc in next_chars:
                if K > 1:
                    new_state = (state_str + nc)[- (K-1):]
                else:
                    new_state = ''
                # Check if previous state length was K-1, then check palindrome
                if len(state_str) == K - 1:
                    substr = state_str + nc
                    if substr == substr[::-1]:
                        continue  # Skip this transition
                # Update the current DP
                curr_dp[new_state] = (curr_dp[new_state] + cnt) % MOD
        prev_dp = curr_dp
    
    ans = sum(prev_dp.values()) % MOD
    print(ans)

if __name__ == "__main__":
    main()