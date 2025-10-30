class Solution:
    def minSizeSubarray(self, nums: List[int], target: int) -> int:
        target -= sum(nums) * (target // sum(nums))
        nums *= 2
        n, ans = len(nums), inf
        for i in range(n):
            sm = 0
            for j in range(i, n):
                sm += nums[j]
                if sm == target:
                    ans = min(ans, j - i + 1)
                elif sm > target:
                    break
        return ans if ans < inf else -1