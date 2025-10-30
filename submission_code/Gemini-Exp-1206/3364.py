class Solution:
    def minimumValueSum(self, nums: List[int], andValues: List[int]) -> int:
        n = len(nums)
        m = len(andValues)
        dp = {}

        def solve(idx, and_idx):
            if idx == n:
                return 0 if and_idx == m else float('inf')
            if and_idx == m:
                return float('inf')

            if (idx, and_idx) in dp:
                return dp[(idx, and_idx)]

            ans = float('inf')
            curr_and = nums[idx]
            for i in range(idx, n):
                curr_and &= nums[i]
                if curr_and == andValues[and_idx]:
                    ans = min(ans, nums[i] + solve(i + 1, and_idx + 1))

            dp[(idx, and_idx)] = ans
            return ans

        result = solve(0, 0)
        return result if result != float('inf') else -1