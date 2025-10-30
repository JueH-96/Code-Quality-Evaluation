class Solution:
    def findMinimumOperations(self, s1: str, s2: str, s3: str) -> int:
        if len(s1) + len(s2) != len(s3):
            return -1
        count1, count2 = [0]*26, [0]*26
        for i in range(len(s1)):
            count1[ord(s1[i]) - ord('a')] += 1
        for i in range(len(s2)):
            count2[ord(s2[i]) - ord('a')] += 1
        for i in range(len(s3)):
            count1[ord(s3[i]) - ord('a')] += 1
        dp = [[0]*2000 for _ in range(2000)]
        for i in range(len(s1)+1):
            dp[i][0] = i
        for j in range(len(s2)+1):
            dp[0][j] = j
        for i in range(1, len(s1)+1):
            for j in range(1, len(s2)+1):
                if s1[i-1] == s2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = min(dp[i-1][j]+1, dp[i][j-1]+1)
        res = float('inf')
        for i in range(26):
            if count1[i] < count2[i]:
                return -1
            res = min(res, dp[count1[i]][count2[i]])
        return res