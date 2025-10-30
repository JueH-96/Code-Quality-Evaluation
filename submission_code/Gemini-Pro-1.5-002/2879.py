class Solution:
    def minimumChanges(self, s: str, k: int) -> int:
        n = len(s)
        dp = [[float('inf')] * (k + 1) for _ in range(n + 1)]
        dp[0][0] = 0

        for i in range(1, n + 1):
            for j in range(1, min(i, k) + 1):
                for l in range(1, i + 1):
                    sub = s[i-l:i]
                    cost = 0
                    
                    for d in range(1, len(sub)):
                        if len(sub) % d == 0:
                            for start in range(d):
                                chars = {}
                                for idx in range(start, len(sub), d):
                                    chars[sub[idx]] = chars.get(sub[idx], 0) + 1
                                
                                max_freq = 0
                                for freq in chars.values():
                                    max_freq = max(max_freq, freq)
                                cost = min(cost, len(sub) // d * d - max_freq if cost==0 else cost)
                    if cost == float('inf'):
                        for d in range(1, len(sub)):
                            if len(sub) % d == 0:
                                cost = 0
                                for st in range(d):
                                    counts = {}
                                    for idx in range(st, len(sub), d):
                                        counts[sub[idx]] = counts.get(sub[idx], 0) + 1
                                    mx_cnt = 0
                                    for cnt in counts.values():
                                        mx_cnt = max(mx_cnt, cnt)
                                    cost = min(len(sub) - mx_cnt, cost if cost != 0 else len(sub) - mx_cnt)
                    
                    
                    dp[i][j] = min(dp[i][j], dp[i-l][j-1] + cost)
        
        return dp[n][k]