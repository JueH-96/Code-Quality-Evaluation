class Solution:
    def missingInteger(self, nums: List[int]) -> int:
        # Step 1: Find the sum of the longest sequential prefix
        sum_seq = nums[0]
        for i in range(1, len(nums)):
            if nums[i] == nums[i-1] + 1:
                sum_seq += nums[i]
            else:
                break
        
        # Step 2: Find the smallest missing integer x >= sum_seq
        num_set = set(nums)
        x = sum_seq
        while x in num_set:
            x += 1
        return x