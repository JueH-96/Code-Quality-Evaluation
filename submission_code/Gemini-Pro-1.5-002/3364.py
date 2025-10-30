class Solution:
    def minimumValueSum(self, nums: List[int], andValues: List[int]) -> int:
        n = len(nums)
        m = len(andValues)

        @lru_cache(None)
        def solve(idx, k):
            if k == m:
                if idx == n:
                    return 0
                else:
                    return float('inf')

            if idx == n:
                return float('inf')
            
            res = float('inf')
            curr_and = nums[idx]
            for i in range(idx, n):
                curr_and &= nums[i]
                if curr_and == andValues[k]:
                    res = min(res, nums[i] + solve(i + 1, k + 1))
            return res

        ans = solve(0, 0)
        return ans if ans != float('inf') else -1