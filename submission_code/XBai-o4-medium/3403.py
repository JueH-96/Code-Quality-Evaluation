class Solution:
    def minimumSubstringsInPartition(self, s: str) -> int:
        n = len(s)
        dp = [float('inf')] * (n + 1)
        dp[0] = 0
        
        for i in range(1, n + 1):
            count = [0] * 26
            for j in range(i-1, -1, -1):
                char = s[j]
                index = ord(char) - ord('a')
                count[index] += 1
                
                # Check if the current substring s[j:i] is balanced
                first = None
                balanced = True
                for k in range(26):
                    if count[k] == 0:
                        continue
                    if first is None:
                        first = count[k]
                    else:
                        if count[k] != first:
                            balanced = False
                            break
                if balanced:
                    if dp[j] + 1 < dp[i]:
                        dp[i] = dp[j] + 1
        
        return dp[n]