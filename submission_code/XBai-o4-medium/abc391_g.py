import sys
from collections import defaultdict

MOD = 998244353

def main():
    N, M = map(int, sys.stdin.readline().split())
    S = sys.stdin.readline().strip()
    pre_S = [ord(c) - ord('a') for c in S]
    current_dp = defaultdict(int)
    initial_state = tuple([0]*(N+1))
    current_dp[initial_state] = 1
    
    for _ in range(M):
        next_dp = defaultdict(int)
        for state, cnt in current_dp.items():
            for c in range(26):
                new_row = [0]*(N+1)
                for i in range(1, N+1):
                    if pre_S[i-1] == c:
                        new_val = state[i-1] + 1
                    else:
                        new_val = state[i]
                    new_row[i] = max(new_row[i-1], new_val)
                new_state = tuple(new_row)
                next_dp[new_state] = (next_dp[new_state] + cnt) % MOD
        current_dp = next_dp
    
    ans = [0]*(N+1)
    for state, cnt in current_dp.items():
        l = state[N]
        ans[l] = (ans[l] + cnt) % MOD
    
    print(' '.join(map(str, ans)))

if __name__ == '__main__':
    main()