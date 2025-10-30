from typing import List

class Solution:
  """
  Solves the problem of determining if an array can be sorted by swapping adjacent elements
  that have the same number of set bits.
  """
  def canSortArray(self, nums: List[int]) -> bool:
    """
    Checks if the array `nums` can be sorted into non-decreasing order using the allowed adjacent swaps.
    The core idea is based on partitioning the array into contiguous blocks where elements within each block
    have the same number of set bits. Elements can be freely rearranged within their block (effectively sorted),
    but cannot move across block boundaries. The array is sortable if and only if the maximum element
    of any block is less than or equal to the minimum element of the immediately following block.

    Args:
        nums: A list of positive integers.

    Returns:
        True if the array can be sorted according to the rules, False otherwise.
    """
    
    def countSetBits(n: int) -> int:
        """
        Helper function to count the number of set bits (1s) in the binary representation of a positive integer n.
        Uses Python's built-in functionality `bin(n).count('1')`.
        For Python 3.10+, `n.bit_count()` could be used for potential performance benefits.
        """
        # Example: bin(8) -> '0b1000', count('1') -> 1
        # Example: bin(15) -> '0b1111', count('1') -> 4
        return bin(n).count('1')

    n = len(nums)
    # An array with 0 or 1 element is always sorted by definition.
    if n <= 1:
        return True

    # Stores the maximum value encountered in the previously processed block.
    # Initialized to 0. Since all elements in nums are positive (>= 1), 
    # 0 is guaranteed to be less than or equal to the minimum element of the first block.
    prev_max = 0 
    
    # `i` serves as the main pointer iterating through the array, marking the start of each new block.
    i = 0
    while i < n:
        # Calculate the number of set bits for the first element of the current block (nums[i]).
        num_bits = countSetBits(nums[i])
        
        # Initialize the minimum and maximum values found within the current block.
        # Start with the first element of the block.
        current_min = nums[i]
        current_max = nums[i]
        
        # `current_idx` is used to scan through the elements belonging to the current block.
        # It starts from the element right after the block's first element (nums[i]).
        current_idx = i + 1
        # Continue scanning as long as we are within the array bounds (`current_idx < n`) and 
        # the element `nums[current_idx]` has the same number of set bits (`num_bits`) 
        # as the elements already identified for this block.
        while current_idx < n and countSetBits(nums[current_idx]) == num_bits:
            # Update the minimum and maximum values encountered within this block so far.
            current_min = min(current_min, nums[current_idx])
            current_max = max(current_max, nums[current_idx])
            # Move to the next element to check if it belongs to the current block.
            current_idx += 1
        
        # Check the critical condition for sortability:
        # The maximum value of the *previous* block (`prev_max`) must not be greater than
        # the minimum value of the *current* block (`current_min`).
        # If this condition fails (`prev_max > current_min`), it signifies an inversion
        # between elements that cannot be resolved through allowed swaps because they belong
        # to different blocks. Thus, the array cannot be sorted.
        if prev_max > current_min:
            return False
        
        # Update `prev_max` to the maximum value found in the *current* block.
        # This value will be used in the next iteration to check against the minimum of the next block.
        prev_max = current_max
        
        # Move the main pointer `i` to the start of the next block.
        # `current_idx` now points to the index immediately following the end of the current block
        # (or equals `n` if the current block extended to the end of the array).
        i = current_idx 

    # If the loop completes without returning False, it means the condition `prev_max <= current_min`
    # held true for all adjacent blocks. This implies that after sorting elements within each block,
    # the entire array will be in non-decreasing order.
    return True