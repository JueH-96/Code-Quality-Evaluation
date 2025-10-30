class Solution:
    def maximumTotalDamage(self, power: List[int]) -> int:
        n = len(power)
        power.sort()
        dp = {}

        def solve(index):
            if index >= n:
                return 0
            if index in dp:
                return dp[index]

            ans = 0
            # Option 1: Don't include the current spell
            ans = max(ans, solve(index + 1))

            # Option 2: Include the current spell
            next_index = index + 1
            while next_index < n and power[next_index] <= power[index] + 2:
                next_index += 1
            ans = max(ans, power[index] + solve(next_index))

            dp[index] = ans
            return ans

        return solve(0)