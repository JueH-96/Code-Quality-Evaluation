import bisect
from typing import List

class Solution:
  """
  Solves the maximum frequency problem after operations.
  """
  def maxFrequency(self, nums: List[int], k: int, numOperations: int) -> int:
    """
    Calculates the maximum possible frequency of any element after performing at most numOperations operations.
    Each operation allows changing nums[i] by adding a value in [-k, k], effectively changing nums[i] to any value
    in the range [nums[i] - k, nums[i] + k].

    Args:
        nums: The input list of integers.
        k: The maximum absolute change allowed in one operation.
        numOperations: The maximum number of operations allowed.

    Returns:
        The maximum frequency achievable for any element.
    """
        
    n = len(nums)
    if n == 0:
        # If the input list is empty, the maximum frequency is 0.
        return 0
        
    # Sort the array to enable efficient range queries using binary search.
    # Sorting takes O(N log N) time.
    nums.sort()
        
    # Generate candidate target values.
    # We want to find a target value T and a set of indices I such that for all i in I,
    # nums[i] can be changed to T (i.e., |nums[i] - T| <= k), and the number of elements
    # in I that are not originally T is at most numOperations.
    # The maximum size of such set I is the answer.
    # It can be shown that the optimal target value T that maximizes the frequency
    # must be among the values x, x - k, or x + k for some x in nums.
    # This is because the set of elements potentially convertible to T, S_T = {i | |nums[i] - T| <= k},
    # changes its composition only when T crosses a boundary point nums[i] +/- k.
    # Similarly, the count of elements already equal to T, E_T = {i | nums[i] == T}, changes when T crosses a value nums[i].
    # The achievable frequency depends on the size of S_T and E_T.
    # Thus, checking values of the form nums[i] and nums[i] +/- k is sufficient.
    candidates = set()
    for x in nums:
        candidates.add(x)
        # Add boundary points related to k. If k=0, x-k and x+k are the same as x,
        # correctly handled by using a set.
        candidates.add(x - k)
        candidates.add(x + k)
            
    # Convert the set of candidates to a sorted list. Sorting is not strictly necessary
    # for correctness but can be good practice and potentially offers minor locality benefits.
    # There are at most 3N candidates. Sorting takes O(N log N) time.
    sorted_candidates = sorted(list(candidates))
        
    max_freq = 0
    # Base case: If the array is not empty, the frequency is at least 1.
    # Any single element can be a group of frequency 1 with 0 operations.
    if n > 0:
       max_freq = 1 

    # Iterate through each candidate target value T.
    # This loop runs O(N) times.
    for T in sorted_candidates:
            
        # For a fixed target T, find the range of indices [L, R] in the sorted nums array
        # such that elements nums[j] are potentially convertible to T.
        # The condition is |nums[j] - T| <= k, which is equivalent to nums[j] in [T - k, T + k].
        # We use binary search (bisect_left, bisect_right) to find this range efficiently.
        # Both bisect_left and bisect_right take O(log N) time.
            
        # Find the first index L such that nums[L] >= T - k.
        L = bisect.bisect_left(nums, T - k)
            
        # Find the first index R_idx such that nums[R_idx] > T + k.
        # The index of the last element <= T + k is R_idx - 1.
        R_idx = bisect.bisect_right(nums, T + k)
        R = R_idx - 1
            
        # If L > R, it means no elements in nums fall within the range [T - k, T + k].
        # So, no elements can be converted to T. Skip this candidate T.
        if L > R:
            continue

        # The size of the set S_T = { indices j such that |nums[j] - T| <= k }
        # corresponds to the number of elements in the range nums[L..R].
        # This is the count of elements potentially convertible to T.
        current_S_size = R - L + 1
            
        # Find the count of elements exactly equal to T within the relevant range [L, R].
        # These elements form the set E_T intersected with the potential candidates S_T.
            
        # Find the range [p_L, p_R] of indices where nums[j] == T in the whole sorted array.
        # This also takes O(log N) time using binary search.
        p_L = bisect.bisect_left(nums, T)
        p_R_idx = bisect.bisect_right(nums, T)
        p_R = p_R_idx - 1

        count_E_T = 0
        # Check if T exists in nums (p_L <= p_R indicates T is present)
        if p_L <= p_R:
             # Calculate the intersection of the range of potential candidates [L, R]
             # and the range of elements exactly equal to T [p_L, p_R].
             # The intersection represents elements that are potentially convertible to T *and* are already equal to T.
             intersect_start = max(L, p_L)
             intersect_end = min(R, p_R)
                 
             # If the intersection range is valid (start index <= end index), calculate its size.
             if intersect_start <= intersect_end:
                 count_E_T = intersect_end - intersect_start + 1

        # Calculate the number of elements that are potentially convertible to T
        # but are not already equal to T. These are the elements that would require an operation.
        # This count is |N_T| = |S_T| - |E_T|.
        count_N_T = current_S_size - count_E_T
            
        # Calculate the achievable frequency for the current target T.
        # We can always include the elements already equal to T (count_E_T).
        # From the elements that need modification (count_N_T), we can include at most numOperations.
        current_freq = count_E_T + min(count_N_T, numOperations)
            
        # Update the overall maximum frequency found so far across all candidate targets.
        max_freq = max(max_freq, current_freq)
            
    # The overall time complexity is dominated by sorting and the loop over candidates.
    # O(N log N) + O(N * log N) = O(N log N).
    # The space complexity is O(N) for storing candidates, or potentially O(1) if candidates are processed differently.
    # Current implementation uses O(N) space for the candidates set and sorted list.
    return max_freq