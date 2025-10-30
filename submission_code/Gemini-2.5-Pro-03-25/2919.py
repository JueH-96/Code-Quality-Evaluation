import math
from typing import List

class Solution:
  """
  Implements the solution to find the maximum number of increasing groups using a greedy approach.
  """
  def maxIncreasingGroups(self, usageLimits: List[int]) -> int:
    """
    Calculates the maximum number of groups (k) that can be formed satisfying the given conditions:
    1. Group sizes are strictly increasing: |G_1| < |G_2| < ... < |G_k|. This implies |G_j| >= j for minimum sizes.
    2. Each group G_i contains distinct numbers from the set {0, 1, ..., n-1}.
    3. Each number j can be used at most usageLimits[j] times in total across all groups.

    Args:
        usageLimits: A 0-indexed list of integers where usageLimits[i] specifies the maximum
                     number of times the number `i` can be included in groups. The length of the list is n.

    Returns:
        An integer representing the maximum number of groups `k` that can be formed.
    
    The approach is based on a greedy strategy. We sort the usage limits to process elements
    based on their availability (scarcity). We iterate through the sorted limits, maintaining a cumulative
    sum of available element counts (`available_sum`) and the current maximum number of groups `k` that 
    can potentially be formed. At each step `i`, we check if the resources accumulated so far 
    (elements corresponding to sorted indices 0 to `i`) are sufficient to support forming `k+1` groups.
    Sufficiency requires meeting two necessary conditions: having enough distinct elements and having enough
    total element count for the minimum requirements of `k+1` groups. If both conditions are met, we
    increment `k`.
    """
    
    n = len(usageLimits)
    
    # Sort usageLimits in non-decreasing order. This allows us to incrementally consider
    # elements based on their usage limits. The element corresponding to index `i` in the
    # sorted list is conceptually the i-th element type when ordered by availability.
    usageLimits.sort()
    
    k = 0  # Initialize the count of groups potentially formed to 0.
    available_sum = 0 # Initialize the cumulative sum of usage limits considered so far.

    # Iterate through the sorted usage limits. `i` ranges from 0 to n-1.
    for i in range(n):
        # Accumulate the usage limit of the current element (at sorted index i).
        # This `available_sum` represents the total count supply from the first i+1 elements
        # with the smallest usage limits.
        available_sum += usageLimits[i]
        
        # Check if it's possible to form k+1 groups using the elements corresponding to indices 0..i.
        
        # Condition 1: Distinct Elements Constraint Check
        # To form k+1 groups, the largest group G_{k+1} must have size |G_{k+1}| >= k+1.
        # This requires at least k+1 distinct element types.
        # Considering elements corresponding to sorted indices 0 to i, we have i+1 distinct element types available.
        # Thus, a necessary condition is i+1 >= k+1, which simplifies to i >= k.
        
        # Condition 2: Total Element Count Constraint Check
        # The minimum total number of elements required to form k+1 groups with the smallest possible
        # increasing sizes (1, 2, ..., k+1) is the sum of the first k+1 positive integers.
        # This sum is T_{k+1} = 1 + 2 + ... + (k+1) = (k+1)*(k+2)/2.
        # The total count of elements available from the usage limits of elements 0..i is `available_sum`.
        # Thus, a necessary condition is `available_sum >= T_{k+1}`.
        
        # Calculate the minimum required total sum for k+1 groups.
        # Python's arbitrary precision integers handle potentially large sums without overflow.
        # Integer division // is used to ensure the result is an integer.
        required_sum_for_k_plus_1 = (k + 1) * (k + 2) // 2
        
        # The core greedy step: If both necessary conditions are satisfied, we assume it's
        # possible to form one more group and increment k. The proof of correctness for this greedy
        # approach relies on showing that if these conditions are met, resources can be allocated appropriately.
        # Primarily, if the algorithm determines k groups are possible, it implies these minimal checks hold.
        # If the checks fail for k+1 groups throughout, it implies k+1 groups are impossible.
        if i >= k and available_sum >= required_sum_for_k_plus_1:
             # Increment k, signifying that we can potentially form k+1 groups.
             k += 1
             
    # After iterating through all elements, the final value of k is the maximum number
    # of groups that satisfy the conditions according to this greedy strategy.
    return k