import math
from typing import List

class Solution:
  """
  Finds the maximum number of subarrays in a split of `nums` such that the sum of scores
  (bitwise AND of elements in subarray) is minimized.
  """
  def maxSubarrays(self, nums: List[int]) -> int:
    """
    Calculates the maximum number of subarrays for a minimum score split.

    Args:
        nums: A list of non-negative integers.

    Returns:
        The maximum number of subarrays.
    """
    
    # Calculate the bitwise AND of all elements in the array.
    # Let this be s_total. The minimum possible sum of scores for any partition is s_total.
    # A partition into k subarrays with scores s_1, ..., s_k has total score Sum(s_i).
    # Since s_i = AND(elements in subarray i) >= AND(all elements) = s_total,
    # Sum(s_i) >= k * s_total.
    # The minimum sum is achieved when k=1 (the whole array), where the sum is s_total.
    
    s_total = -1 # Initialize with all bits set to 1. In Python's two's complement representation, -1 has all bits set to 1.
                 # For any non-negative integer x, -1 & x == x. So -1 acts as an identity element for AND accumulation.
    for x in nums:
        s_total &= x
        # Optimization check: if s_total becomes 0 at any point, the final s_total must be 0.
        # However, we need to iterate through all elements anyway for the greedy part if s_total=0.
        
    # Case 1: The minimum possible score sum (s_total) is greater than 0.
    # As derived above, Sum(s_i) >= k * s_total.
    # To achieve the minimum sum s_total, we must have k * s_total <= s_total.
    # Since s_total > 0, this implies k <= 1.
    # Since we must have at least one subarray (k >= 1), the only possibility is k = 1.
    # This corresponds to the partition consisting of the entire array as a single subarray.
    # Therefore, the maximum number of subarrays is 1.
    if s_total > 0:
        return 1
        
    # Case 2: The minimum possible score sum (s_total) is 0.
    # To achieve the minimum sum 0, we need a partition into k subarrays such that Sum(s_i) = 0.
    # Since all elements are non-negative, the scores s_i are also non-negative.
    # Thus, Sum(s_i) = 0 implies s_i = 0 for all i = 1, ..., k.
    # We need to find a partition into the maximum possible number of subarrays (k),
    # such that the score of each subarray is 0.
    
    # We use a greedy strategy: Iterate through the array and form subarrays.
    # Start a new subarray. Keep track of the running bitwise AND of its elements.
    # As soon as the running AND becomes 0, finalize this subarray and start a new one.
    # This strategy maximizes the number of subarrays because each subarray found is the shortest possible
    # segment starting from its beginning index that achieves a score of 0.
    
    count = 0
    # current_and tracks the running bitwise AND for the current segment being formed.
    current_and = -1 # Initialize with all bits 1.
    
    for x in nums:
        current_and &= x
        # When the running AND becomes 0, it means the segment from the start
        # of the current accumulation up to the current element `x` has a score of 0.
        if current_and == 0:
            # Increment the count of subarrays found with score 0.
            count += 1
            # Reset the running AND to start accumulating for the next potential subarray.
            current_and = -1 
            
    # Since s_total is 0, the bitwise AND of the entire array is 0.
    # This guarantees that the `current_and` variable, accumulating the AND from the start of a segment,
    # must eventually become 0 at least once over the entire array traversal (if not multiple times).
    # For example, even if it only becomes 0 at the very last element, we would count 1 segment.
    # Hence, `count` will be at least 1 if s_total is 0.
    # The greedy approach correctly identifies the maximum possible number of 0-score subarrays
    # that form a partition of the prefix of the array. If the entire array is covered, this `count` is the answer.
    # If the greedy process leaves a suffix with non-zero AND score, it means no partition exists with >1 segments all having score 0.
    # In this scenario, the only partition yielding score 0 is the entire array itself (k=1). The greedy count calculation
    # correctly yields 1 in such cases as well (e.g., [1, 2, 3]).
    return count