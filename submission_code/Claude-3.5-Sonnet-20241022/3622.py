class Solution:
    def maxFrequency(self, nums: List[int], k: int, numOperations: int) -> int:
        n = len(nums)
        # If we can modify all numbers, return the length of array
        if numOperations >= n:
            return n
        
        # Sort the array to make it easier to process
        nums.sort()
        
        # Initialize result with 1 (minimum possible frequency)
        result = 1
        
        # For each unique number in nums
        i = 0
        while i < n:
            # Count how many times current number appears
            curr_count = 1
            while i + 1 < n and nums[i] == nums[i + 1]:
                curr_count += 1
                i += 1
            
            # Try to make other numbers equal to nums[i]
            operations_left = numOperations
            j = i + 1
            
            # Count how many numbers we can make equal to nums[i]
            while j < n and operations_left > 0:
                if nums[j] - nums[i] <= k:
                    curr_count += 1
                    operations_left -= 1
                else:
                    break
                j += 1
            
            # Try numbers before i
            j = i - 1
            while j >= 0 and operations_left > 0:
                if nums[i] - nums[j] <= k:
                    curr_count += 1
                    operations_left -= 1
                else:
                    break
                j -= 1
            
            result = max(result, curr_count)
            i += 1
        
        return result