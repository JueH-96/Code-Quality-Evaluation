class Solution:
    def maximumTripletValue(self, nums: List[int]) -> int:
        n = len(nums)
        max_diff = [0] * n
        max_so_far = nums[0]
        
        for i in range(1, n):
            max_diff[i] = max(max_diff[i-1], max_so_far - nums[i])
            max_so_far = max(max_so_far, nums[i])
        
        max_value = 0
        for k in range(2, n):
            max_value = max(max_value, max_diff[k-1] * nums[k])
        
        return max_value