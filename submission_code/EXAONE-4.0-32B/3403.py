class Solution:
    def minimumSubstringsInPartition(self, s: str) -> int:
        n = len(s)
        dp = [float('inf')] * (n + 1)
        dp[0] = 0
        
        for i in range(1, n + 1):
            freq = [0] * 26
            distinct_count = 0
            current_max = 0
            
            for j in range(i - 1, -1, -1):
                char = s[j]
                idx = ord(char) - ord('a')
                old_count = freq[idx]
                new_count = old_count + 1
                freq[idx] = new_count
                
                if old_count == 0:
                    distinct_count += 1
                
                if new_count > current_max:
                    current_max = new_count
                
                L = i - j
                if distinct_count > 0 and L % distinct_count == 0:
                    k0 = L // distinct_count
                    if current_max == k0:
                        if dp[j] + 1 < dp[i]:
                            dp[i] = dp[j] + 1
        
        return dp[n]