mod = 998244353

import sys

def main():
    data = sys.stdin.read().splitlines()
    if not data:
        return
    parts = data[0].split()
    N = int(parts[0])
    M = int(parts[1])
    S = data[1].strip()
    
    total_states = 1 << N
    trans_table = [[0] * 26 for _ in range(total_states)]
    
    for mask in range(total_states):
        f = [0] * (N + 1)
        for j in range(1, N + 1):
            if mask & (1 << (j - 1)):
                f[j] = f[j - 1] + 1
            else:
                f[j] = f[j - 1]
                
        for c_int in range(26):
            c_char = chr(ord('a') + c_int)
            g = [0] * (N + 1)
            next_mask = 0
            for j in range(1, N + 1):
                op1 = f[j]
                op2 = g[j - 1]
                op3 = f[j - 1] + (1 if S[j - 1] == c_char else 0)
                g[j] = max(op1, op2, op3)
                if g[j] > g[j - 1]:
                    next_mask |= (1 << (j - 1))
            trans_table[mask][c_int] = next_mask
            
    dp = [0] * total_states
    dp[0] = 1
    
    for step in range(M):
        new_dp = [0] * total_states
        for mask in range(total_states):
            count_here = dp[mask]
            if count_here == 0:
                continue
            for c_int in range(26):
                next_mask = trans_table[mask][c_int]
                new_dp[next_mask] = (new_dp[next_mask] + count_here) % mod
        dp = new_dp
        
    ans = [0] * (N + 1)
    for mask in range(total_states):
        k = bin(mask).count('1')
        ans[k] = (ans[k] + dp[mask]) % mod
        
    print(" ".join(map(str, ans)))
    
if __name__ == "__main__":
    main()