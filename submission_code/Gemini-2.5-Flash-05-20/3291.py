import collections

class Solution:
    def canSortArray(self, nums: List[int]) -> bool:
        n = len(nums)
        if n <= 1:
            return True

        # Step 1: Precompute popcounts for all numbers in nums.
        # int.bit_count() is used for efficiency.
        # For older Python versions, bin(x).count('1') can be used.
        popcounts = [x.bit_count() for x in nums]

        # Step 2: Iterate through the array to find contiguous blocks
        # of numbers with the same popcount. Sort each block in-place.
        i = 0
        while i < n:
            # Find the start of the current block
            current_popcount = popcounts[i]
            
            # Find the end of the current block (exclusive index j)
            # All elements from nums[i] to nums[j-1] will have current_popcount
            j = i
            while j < n and popcounts[j] == current_popcount:
                j += 1
            
            # Extract the current block, sort it, and place it back.
            # Python's list slicing nums[i:j] creates a new list (copy),
            # which is then sorted and assigned back to the original slice,
            # effectively sorting in-place conceptually.
            sub_array = nums[i:j]
            sub_array.sort()
            
            # Assign the sorted sub_array back to the original nums list
            nums[i:j] = sub_array
            
            # Move 'i' to the start of the next block
            i = j
        
        # Step 3: After sorting all eligible blocks, check if the entire array is sorted.
        for k in range(n - 1):
            if nums[k] > nums[k+1]:
                return False # Array is not sorted
                
        return True # Array is sorted