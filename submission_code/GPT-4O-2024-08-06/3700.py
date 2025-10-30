class Solution:
    def subsequencesWithMiddleMode(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        n = len(nums)
        count = 0
        
        # Iterate over all possible middle elements
        for i in range(2, n - 2):
            middle = nums[i]
            
            # Count occurrences of middle in the left and right parts
            left_count = 0
            right_count = 0
            
            # Count occurrences of middle to the left of i
            for j in range(i):
                if nums[j] == middle:
                    left_count += 1
            
            # Count occurrences of middle to the right of i
            for j in range(i + 1, n):
                if nums[j] == middle:
                    right_count += 1
            
            # Calculate the number of valid subsequences
            # We need at least one occurrence of middle on both sides
            if left_count > 0 and right_count > 0:
                # Choose one occurrence from left and one from right
                count += left_count * right_count
                count %= MOD
        
        return count