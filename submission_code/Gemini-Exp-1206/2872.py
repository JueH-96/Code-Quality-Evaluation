class Solution:
    def maxArrayValue(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 1:
            return nums[0]
        
        ans = 0
        cur_sum = 0
        for i in range(n - 1, -1, -1):
            if i == n - 1:
                cur_sum = nums[i]
            elif nums[i] <= cur_sum:
                cur_sum += nums[i]
            else:
                cur_sum = nums[i]
            ans = max(ans, cur_sum)
        
        return ans