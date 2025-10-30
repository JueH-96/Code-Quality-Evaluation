class Solution:
    def maximumTripletValue(self, nums: List[int]) -> int:
        max_value = 0
        max_num = 0  # max nums[i] for i < current index
        max_diff = 0  # max (nums[i] - nums[j]) for i < j < current index
        
        for k in range(len(nums)):
            if k > 1:  # we can form a triplet
                max_value = max(max_value, max_diff * nums[k])
            
            if k > 0:  # we can form a difference
                max_diff = max(max_diff, max_num - nums[k])
            
            max_num = max(max_num, nums[k])
        
        return max_value