from typing import List

# Helper function to count set bits
# Placed outside the class as it's a utility function
def countSetBits(n: int) -> int:
    """Counts the number of set bits (1s) in the binary representation of a positive integer."""
    # Using Python's built-in binary representation string method is simple and efficient for small numbers.
    # For the given constraints (1 <= nums[i] <= 2^8), this is sufficient.
    return bin(n).count('1')

class Solution:
    def canSortArray(self, nums: List[int]) -> bool:
        n = len(nums)
        
        # Calculate the number of set bits for each number in the input array.
        # This list helps identify contiguous segments with the same bit count.
        bit_counts = [countSetBits(x) for x in nums]
        
        # Create the target sorted version of the original array.
        # This is what we need to be able to reach through allowed operations.
        sorted_nums = sorted(nums)
        
        # Use a pointer 'i' to mark the start of the current contiguous segment
        # of elements having the same number of set bits in the original array 'nums'.
        i = 0
        while i < n:
            # Find the end of the current segment.
            # 'j' is a pointer that iterates through the segment starting from 'i'.
            # The segment ends just before the index where the bit count changes or at the end of the array.
            j = i
            current_bit_count = bit_counts[i]
            while j < n and bit_counts[j] == current_bit_count:
                j += 1
            # The current segment in the original array 'nums' is from index 'i' up to (but not including) 'j'.
            # This corresponds to the segment in the target sorted array 'sorted_nums' from index 'i' up to 'j'.
            
            # Extract the values within the current segment from the original array.
            segment_nums = nums[i:j]
            
            # Sort these extracted values.
            # Since elements within a contiguous block of the same bit count can be arbitrarily permuted
            # relative to each other (using adjacent swaps), they *can* be sorted internally.
            segment_nums_sorted = sorted(segment_nums)
            
            # Extract the corresponding values from the fully sorted target array.
            # These are the values that *must* occupy the indices [i, j-1] in the final sorted array.
            segment_sorted_nums = sorted_nums[i:j]
            
            # Compare the sorted segment from the original array's values
            # with the values from the target sorted array's corresponding segment.
            # If these two lists are not identical, it means the set of values present
            # in the original array's segment (indices i to j-1) is not the same as the
            # set of values required in the target sorted array's segment (indices i to j-1).
            # Since elements cannot move across the boundaries defined by different bit counts,
            # the values are effectively trapped within their initial contiguous bit-count blocks
            # relative to elements outside the block. If the wrong values are in a block,
            # sorting the entire array is impossible.
            if segment_nums_sorted != segment_sorted_nums:
                return False
            
            # Move the starting index 'i' to the beginning of the next segment.
            # 'j' is already positioned at the start of the next potential segment
            # or is equal to 'n' if the current segment was the last one.
            i = j
            
        # If the comparison passed for all contiguous segments in the original array,
        # it means that for every block of elements that are "stuck" together
        # (because they cannot swap with elements outside the block), the values
        # within that block in the original array are exactly the values needed
        # in the corresponding positions in the final sorted array. Since elements
        # within such a block can be rearranged arbitrarily, they can be sorted
        # to match the required order. Thus, the array can be sorted.
        return True