import sys
from collections import defaultdict

MOD = 998244353

def main():
    N, M = map(int, sys.stdin.readline().split())
    S = sys.stdin.readline().strip()
    
    def compute_f(k_val):
        if k_val >= N:
            return pow(26, M, MOD)
        dp = defaultdict(int)
        initial = tuple([0] * (N + 1))
        dp[initial] = 1
        for _ in range(M):
            new_dp = defaultdict(int)
            for state in dp:
                count = dp[state]
                for c in 'abcdefghijklmnopqrstuvwxyz':
                    new_row = [0] * (N + 1)
                    for i in range(1, N + 1):
                        new_val = max(state[i], new_row[i-1])
                        if S[i-1] == c:
                            new_val = max(new_val, state[i-1] + 1)
                        new_row[i] = new_val
                    new_state = tuple(new_row)
                    if new_row[N] <= k_val:
                        new_dp[new_state] = (new_dp[new_state] + count) % MOD
            dp = new_dp
        return sum(dp.values()) % MOD
    
    f = [0] * (N + 2)
    for k in range(N + 1):
        f[k] = compute_f(k)
    
    ans = [0] * (N + 1)
    ans[0] = f[0]
    for k in range(1, N + 1):
        ans[k] = (f[k] - f[k-1]) % MOD
    
    print(' '.join(map(str, ans)))

if __name__ == '__main__':
    main()