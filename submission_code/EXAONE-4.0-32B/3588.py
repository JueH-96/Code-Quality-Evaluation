class Solution:
    def countWinningSequences(self, s: str) -> int:
        mod = 10**9 + 7
        n = len(s)
        
        def get_delta(a, b):
            if a == 'F':
                if b == 'W': return 1
                elif b == 'E': return -1
                else: return 0
            elif a == 'W':
                if b == 'E': return 1
                elif b == 'F': return -1
                else: return 0
            else:
                if b == 'F': return 1
                elif b == 'W': return -1
                else: return 0
        
        dp = [[0] * (2*n+1) for _ in range(3)]
        creatures = ['F', 'W', 'E']
        
        for idx, b in enumerate(creatures):
            d_val = get_delta(s[0], b)
            diff_index = d_val + n
            if 0 <= diff_index < 2*n+1:
                dp[idx][diff_index] = (dp[idx][diff_index] + 1) % mod
        
        for i in range(1, n):
            new_dp = [[0] * (2*n+1) for _ in range(3)]
            for last in range(3):
                for diff_idx in range(2*n+1):
                    count = dp[last][diff_idx]
                    if count == 0:
                        continue
                    current_diff = diff_idx - n
                    for nxt_idx in range(3):
                        if nxt_idx == last:
                            continue
                        d_val = get_delta(s[i], creatures[nxt_idx])
                        new_diff = current_diff + d_val
                        new_idx = new_diff + n
                        if 0 <= new_idx < 2*n+1:
                            new_dp[nxt_idx][new_idx] = (new_dp[nxt_idx][new_idx] + count) % mod
            dp = new_dp
        
        total = 0
        for last in range(3):
            for diff_idx in range(n+1, 2*n+1):
                total = (total + dp[last][diff_idx]) % mod
        
        return total