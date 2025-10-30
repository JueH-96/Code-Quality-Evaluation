import collections
from typing import List

class Solution:
  """
  Finds the maximum length of a subset of nums that can form a sequence 
  following the pattern [x, x^2, x^4, ..., x^(2^p), ..., x^4, x^2, x].
  """
  def maximumLength(self, nums: List[int]) -> int:
    """
    Calculates the maximum length subset based on the specified pattern.

    The pattern is a sequence starting with x, repeatedly squaring to x^(2^p),
    and then reversing the process back to x. The total length is 2p+1.
    To form such a sequence, we need specific counts of elements:
    - at least 1 occurrence of the peak element x^(2^p)
    - at least 2 occurrences of each element x^(2^i) for 0 <= i < p.
    The base case p=0 gives sequence [x] of length 1, requiring count[x] >= 1.

    Args:
      nums: A list of positive integers.

    Returns:
      The maximum possible length of such a subset sequence.
    """
    counts = collections.Counter(nums)
    
    # Handle the special case x = 1.
    # The sequence pattern for x=1 becomes [1, 1, ..., 1].
    # The length must be odd, 2p+1. We need 2p+1 ones.
    # The maximum odd length possible is determined by the count of 1s available.
    count_1 = counts.get(1, 0)
    if count_1 > 0:
        # If count_1 is odd, the max length using only 1s is count_1.
        # If count_1 is even and positive, the max length is count_1 - 1 
        # (we must use an odd number of elements).
        max_len_1 = count_1 if count_1 % 2 == 1 else count_1 - 1
    else:
        # If there are no 1s, the length is 0 for the x=1 case.
        max_len_1 = 0
            
    # Initialize the overall maximum length with the result from the x=1 case.
    overall_max_len = max_len_1
    
    # Define the maximum possible value for elements in nums based on constraints.
    MAX_VAL = 10**9
    # Precompute integer approximation of sqrt(MAX_VAL) for optimization.
    # If curr > sqrt(MAX_VAL), then curr*curr > MAX_VAL.
    SQRT_MAX_VAL = 31622 

    # Keep track of numbers that have already been considered as part of a sequence check
    # starting from one of their roots (e.g., if 4 is checked as part of the sequence starting from 2,
    # we don't need to start a new check from 4). This avoids redundant computations.
    processed = set() 

    # Iterate through each distinct number present in nums as a potential starting value x.
    # Using counts.keys() naturally iterates through distinct values.
    for x in counts:
        
        # Skip x=1 as it's handled separately.
        # Skip if x has already been processed as part of a sequence check initiated from its root.
        if x == 1 or x in processed:
            continue

        # Stores the maximum length found for sequences starting with the current x.
        current_len_for_x = 0
        
        # Base case: A sequence of length 1, [x], is always possible if x exists in nums (count >= 1).
        if counts.get(x, 0) >= 1:
           current_len_for_x = 1
        
        # Check if longer sequences starting with x are possible. This requires count[x] >= 2.
        if counts.get(x, 0) >= 2:
            # Mark x as processed since we are starting a chain check from it.
            # Any sequence starting with x^2, x^4, etc., will be implicitly covered.
            processed.add(x) 

            # k represents the exponent index in the power series: x^(2^k).
            # It starts at k=1 because we begin by checking x^2 = x^(2^1).
            k = 1 
            # curr holds the current element value in the sequence chain, initially x = x^(2^0).
            curr = x 
            
            # Loop to check for sequence extension: x, x^2, x^4, ...
            while True:
                # Optimization: Check if curr is already large enough that its square will exceed MAX_VAL.
                if curr > SQRT_MAX_VAL:
                   # Cannot extend further. The sequence must peak at curr = x^(2^(k-1)).
                   # The length is 2*(k-1) + 1 = 2k - 1.
                   current_len_for_x = max(current_len_for_x, 2*k - 1)
                   break

                # Calculate the next element in the sequence progression.
                next_val = curr * curr
                
                # Mark next_val as processed early. This ensures that if we encounter next_val later 
                # as a potential starting point in the outer loop, we skip it because it's part of this chain.
                processed.add(next_val) 

                # Check constraint: element value must be <= MAX_VAL.
                if next_val > MAX_VAL:
                   # The next element exceeds the maximum allowed value.
                   # The sequence must peak at the current element `curr`. Length is 2k - 1.
                   current_len_for_x = max(current_len_for_x, 2*k - 1)
                   break

                # Get the count of the next element value in nums.
                count_next = counts.get(next_val, 0)

                if count_next == 0:
                    # Cannot include next_val because it's not present in nums.
                    # Sequence must peak at curr. Length = 2k - 1.
                    current_len_for_x = max(current_len_for_x, 2*k - 1)
                    break
                elif count_next == 1:
                    # Can use next_val = x^(2^k) as the peak element.
                    # This requires count[next_val] >= 1.
                    # The full sequence length is 2k + 1.
                    current_len_for_x = max(current_len_for_x, 2*k + 1)
                    break
                else: # count_next >= 2
                    # We have enough counts (>= 2) for next_val. 
                    # This satisfies the requirement for non-peak elements.
                    # We can potentially extend the sequence further.
                    # Update k to reflect the next power level, and update curr to next_val.
                    k += 1
                    curr = next_val
        
        # After checking possibilities for the current x, update the overall maximum length found.
        overall_max_len = max(overall_max_len, current_len_for_x)
            
    # Return the overall maximum length found across all possible starting values x.
    return overall_max_len