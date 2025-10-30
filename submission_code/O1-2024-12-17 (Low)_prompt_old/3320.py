class Solution:
    def maxOperations(self, nums: List[int]) -> int:
        # If there are fewer than 2 elements, no operations can be performed.
        if len(nums) < 2:
            return 0
        
        # The initial score is dictated by the first two elements.
        target_sum = nums[0] + nums[1]
        count = 1  # We can do at least one operation for sure (the first pair).
        
        # Start checking from the next pair onward.
        i = 2
        while i + 1 < len(nums):
            if nums[i] + nums[i + 1] == target_sum:
                count += 1
                i += 2
            else:
                # As soon as we encounter a different sum, break out.
                break
        
        return count