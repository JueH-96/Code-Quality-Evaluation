class Solution:
    def maxPalindromesAfterOperations(self, words: List[str]) -> int:
        n = len(words)
        dp = [0] * (1 << n)
        dp[0] = 1
        odd_count = [0] * (1 << n)
        even_count = [0] * (1 << n)
        for mask in range(1 << n):
            odd_count_temp = 0
            even_count_temp = 0
            for i in range(n):
                if ((mask >> i) & 1):
                    odd_count_temp += 1
                    even_count_temp += (words[i][0] == words[i][-1])
                else:
                    even_count_temp += 1
            odd_count[mask] = odd_count_temp
            even_count[mask] = even_count_temp
            if odd_count_temp and even_count_temp:
                dp[mask] = dp[mask ^ (1 << odd_count_temp - 1)] | (1 << even_count_temp)
        return dp[(1 << n) - 1]