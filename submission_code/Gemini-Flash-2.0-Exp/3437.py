class Solution:
    def maximumTotalDamage(self, power: List[int]) -> int:
        power.sort()
        n = len(power)
        dp = {}

        def solve(index, last_used):
            if index == n:
                return 0

            if (index, last_used) in dp:
                return dp[(index, last_used)]

            # Option 1: Don't use the current spell
            ans = solve(index + 1, last_used)

            # Option 2: Use the current spell
            if last_used is None or abs(power[index] - last_used) > 2:
                ans = max(ans, power[index] + solve(index + 1, power[index]))

            dp[(index, last_used)] = ans
            return ans

        return solve(0, None)