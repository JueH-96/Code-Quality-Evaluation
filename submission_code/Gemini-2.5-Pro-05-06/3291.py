from typing import List # List is used in type hints

class Solution:
    def _get_set_bits_count(self, n: int) -> int:
        """Helper function to count set bits in a positive integer."""
        # bin(n) returns a string like "0b..." for positive integers.
        # e.g., bin(8) is "0b1000", so bin(8).count('1') is 1.
        # This is efficient for small integers like those constrained (<= 256).
        return bin(n).count('1')

    def canSortArray(self, nums: List[int]) -> bool:
        n = len(nums)
        
        # Constraints: 1 <= nums.length <= 100.
        # An array of length 1 is always sorted.
        if n <= 1:
            return True

        i = 0
        while i < n:
            start_index_of_block = i
            
            # Determine the number of set bits for the first element of this block.
            # All elements in this contiguous block must have this same bit count.
            current_block_bit_count = self._get_set_bits_count(nums[i])
            
            # Find the end of the contiguous block.
            # Advance pointer `j` as long as it's within bounds and the element nums[j]
            # has the same bit count as the first element of the block.
            j = i
            while j < n and self._get_set_bits_count(nums[j]) == current_block_bit_count:
                j += 1
            
            # The current block is the slice nums[start_index_of_block : j].
            # Sort this block. Python's `sorted()` function on a slice creates a new sorted list.
            # Assigning this back to the slice `nums[start_index_of_block:j]` modifies `nums` in place.
            # If the block has 0 or 1 elements, `sorted()` handles it correctly (it remains unchanged or is a copy).
            if start_index_of_block < j : # This condition ensures the slice is valid; j > i means at least one element.
                                          # Effectively, `j - start_index_of_block` is the block length.
                                          # If length is 0 or 1, sorting is trivial. sorted() handles this.
                nums[start_index_of_block:j] = sorted(nums[start_index_of_block:j])
            
            # Move the main pointer `i` to the start of the next potential block.
            i = j 
            
        # After all blocks are internally sorted, check if the entire array `nums` is sorted.
        for k in range(n - 1):
            if nums[k] > nums[k+1]:
                # If any element is greater than its successor, the array is not sorted.
                return False
        
        # If the loop completes, all adjacent pairs are in non-decreasing order.
        return True