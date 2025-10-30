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
    
    total_masks = 1 << N
    next_mask_table = [[0] * 26 for _ in range(total_masks)]
    
    for mask in range(total_masks):
        a = [0] * (N + 1)
        for i in range(1, N + 1):
            if mask & (1 << (i - 1)):
                a[i] = a[i - 1] + 1
            else:
                a[i] = a[i - 1]
                
        for c_int in range(26):
            char = chr(ord('a') + c_int)
            b = [0] * (N + 1)
            new_mask = 0
            for i in range(1, N + 1):
                b[i] = max(b[i - 1], a[i])
                if S[i - 1] == char:
                    candidate = a[i - 1] + 1
                    if candidate > b[i]:
                        b[i] = candidate
                if b[i] > b[i - 1]:
                    new_mask |= (1 << (i - 1))
            next_mask_table[mask][c_int] = new_mask

    dp = [0] * total_masks
    dp[0] = 1
    
    for _ in range(M):
        new_dp = [0] * total_masks
        for mask in range(total_masks):
            count = dp[mask]
            if count:
                for c_int in range(26):
                    next_mask = next_mask_table[mask][c_int]
                    new_dp[next_mask] = (new_dp[next_mask] + count) % mod
        dp = new_dp
        
    ans = [0] * (N + 1)
    for mask in range(total_masks):
        k = bin(mask).count('1')
        if k <= N:
            ans[k] = (ans[k] + dp[mask]) % mod
            
    print(" ".join(str(x) for x in ans))

if __name__ == '__main__':
    main()