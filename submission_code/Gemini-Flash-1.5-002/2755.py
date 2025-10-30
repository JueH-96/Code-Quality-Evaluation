class Solution:
    def minExtraChar(self, s: str, dictionary: List[str]) -> int:
        n = len(s)
        dp = [float('inf')] * (n + 1)
        dp[0] = 0
        for i in range(n + 1):
            if dp[i] == float('inf'):
                continue
            for word in dictionary:
                j = i + len(word)
                if j <= n and s[i:j] == word:
                    dp[j] = min(dp[j], dp[i])
                
        
        for i in range(n + 1):
            if dp[i] != float('inf'):
                for j in range(i + 1, n + 1):
                    count = 0
                    for k in range(i, j):
                        is_present = False
                        for word in dictionary:
                            if s[k:k+len(word)] == word and k + len(word) <= j:
                                is_present = True
                                break
                        if not is_present:
                            count +=1
                    dp[j] = min(dp[j], dp[i] + count)

        return dp[n]