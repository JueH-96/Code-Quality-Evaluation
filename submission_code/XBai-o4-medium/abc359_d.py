import sys
from collections import defaultdict

MOD = 998244353

def main():
    N, K = map(int, sys.stdin.readline().split())
    S = sys.stdin.readline().strip()
    
    current_dp = defaultdict(int)
    current_dp[""] = 1
    
    for pos in range(N):
        next_dp = defaultdict(int)
        current_char = S[pos]
        for state in current_dp:
            cnt = current_dp[state]
            if current_char == 'A':
                chars = ['A']
            elif current_char == 'B':
                chars = ['B']
            else:
                chars = ['A', 'B']
            for c in chars:
                new_substring = state + c
                if (pos + 1) >= K:
                    # Check if new_substring is a palindrome
                    if new_substring == new_substring[::-1]:
                        continue
                    new_state = new_substring[1:]
                    next_dp[new_state] = (next_dp[new_state] + cnt) % MOD
                else:
                    next_dp[new_substring] = (next_dp[new_substring] + cnt) % MOD
        current_dp = next_dp
        if not current_dp:
            break  # No possible ways, exit early
    
    print(sum(current_dp.values()) % MOD)

if __name__ == "__main__":
    main()