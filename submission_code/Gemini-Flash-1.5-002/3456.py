class Solution:
    def maximumLength(self, nums: List[int], k: int) -> int:
        n = len(nums)
        dp = {}

        def solve(i, prev, changes):
            if i == n:
                return 0
            if (i, prev, changes) in dp:
                return dp[(i, prev, changes)]

            res = solve(i + 1, prev, changes)  # Skip current element

            if nums[i] == prev:
                res = max(res, 1 + solve(i + 1, nums[i], changes))
            elif changes < k:
                res = max(res, 1 + solve(i + 1, nums[i], changes + 1))

            dp[(i, prev, changes)] = res
            return res

        return solve(0, -1, 0)