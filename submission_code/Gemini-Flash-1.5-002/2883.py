class Solution:
    def minimumBeautifulSubstrings(self, s: str) -> int:
        powers_of_5 = set()
        power = 1
        for _ in range(16):
            binary = bin(power)[2:]
            powers_of_5.add(binary)
            power *= 5

        n = len(s)
        dp = {}
        def solve(index):
            if index == n:
                return 0
            if index in dp:
                return dp[index]
            ans = float('inf')
            for i in range(index + 1, n + 1):
                substring = s[index:i]
                if substring in powers_of_5 and (len(substring) == 1 or substring[0] == '1'):
                    ans = min(ans, 1 + solve(i))
            dp[index] = ans
            return ans

        res = solve(0)
        return res if res != float('inf') else -1