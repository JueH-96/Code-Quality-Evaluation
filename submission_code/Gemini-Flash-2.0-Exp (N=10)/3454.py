class Solution:
    def minimumOperations(self, nums: List[int], target: List[int]) -> int:
        n = len(nums)
        diff = [target[i] - nums[i] for i in range(n)]
        
        ans = 0
        prev_diff = 0
        for d in diff:
            if d != prev_diff:
                ans += abs(d - prev_diff)
                prev_diff = d
        return ans