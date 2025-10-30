class Solution:
    def missingInteger(self, nums: List[int]) -> int:
        # Calculate the sum of the longest sequential prefix
        sum_s = nums[0]
        current_val = nums[0]
        for i in range(1, len(nums)):
            if nums[i] == current_val + 1:
                sum_s += nums[i]
                current_val = nums[i]
            else:
                break
        
        # Find the smallest integer >= sum_s not present in nums
        nums_set = set(nums)
        current_x = sum_s
        while True:
            if current_x not in nums_set:
                return current_x
            current_x += 1