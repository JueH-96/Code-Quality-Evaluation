class Solution:
    def minOperations(self, nums: List[int]) -> int:
        n = len(nums)
        diff = [0] * (n + 3)  # difference array to track flip effects
        flips_in_effect = 0   # how many flips are currently affecting this position
        operations = 0
        
        for i in range(n):
            flips_in_effect += diff[i]  # incorporate flips that start at i
            current_val = nums[i] ^ (flips_in_effect & 1)  # value after applying all flips so far
            
            # If current_val is 0, we need to flip at i (which affects i, i+1, i+2)
            if current_val == 0:
                if i + 2 >= n:  # not enough space to flip
                    return -1
                operations += 1
                flips_in_effect += 1  # we start a new flip here
                diff[i + 3] -= 1      # this flip will end after index i+2
        
        return operations