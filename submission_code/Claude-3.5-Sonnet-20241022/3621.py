class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        # If any number is less than k, it's impossible to make all numbers equal to k
        if any(x < k for x in nums):
            return -1
        
        operations = 0
        while len(set(nums)) > 1:  # Continue until all numbers are equal
            # Find all unique values in nums
            unique_vals = sorted(set(nums))
            
            # If the smallest value is greater than k, we can't reach k
            if min(unique_vals) > k:
                # Find the largest value that appears more than once
                max_repeated = k
                for val in range(k, max(nums) + 1):
                    if nums.count(val) > 1:
                        max_repeated = val
                
                # Find valid h: largest value less than smallest number that makes
                # all values greater than h identical
                h = max_repeated
                for val in unique_vals:
                    if val > h and all(x == val for x in nums if x > h):
                        h = val - 1
                        break
                
                # Apply the operation
                for i in range(len(nums)):
                    if nums[i] > h:
                        nums[i] = h
                operations += 1
            else:
                return -1
            
        # If we've made all numbers equal but they're not k, it's impossible
        if nums[0] != k:
            return -1
            
        return operations