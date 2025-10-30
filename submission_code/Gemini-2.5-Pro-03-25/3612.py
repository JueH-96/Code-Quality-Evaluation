import math # Not needed, but good practice to remove unused imports
import collections # Not needed
from typing import List

class Solution:
  """
  Solves the problem of finding two adjacent strictly increasing subarrays of length k.
  """
  def hasIncreasingSubarrays(self, nums: List[int], k: int) -> bool:
    """
    Determines if there exist two adjacent subarrays of length k 
    such that both subarrays are strictly increasing.

    Args:
        nums: A list of integers.
        k: The length of the subarrays.

    Returns:
        True if such adjacent strictly increasing subarrays exist, False otherwise.
    
    Constraints ensure:
        2 <= nums.length <= 100
        1 < 2 * k <= nums.length  (implies k >= 1 and n >= 2*k)
        -1000 <= nums[i] <= 1000
    """
    n = len(nums)

    # Optimization for k=1:
    # If k=1, any subarray of length 1 is strictly increasing by definition.
    # The problem asks for two *adjacent* subarrays of length k=1 that are both strictly increasing.
    # Let the first subarray start at index 'a'. It is nums[a:a+1] = [nums[a]].
    # The second, adjacent subarray starts at index b = a + k = a + 1. It is nums[a+1:a+1+1] = [nums[a+1]].
    # Both [nums[a]] and [nums[a+1]] are strictly increasing subarrays of length 1.
    # We just need to ensure that we can find such an index 'a'.
    # The second subarray ends at index (a+1) + k - 1 = a + 1. This index must be < n. So a+1 < n, or a < n-1.
    # Since the constraint 1 < 2*k <= n implies n >= 2*1 = 2 when k=1, we know n >= 2.
    # If n >= 2, we can always choose a = 0. The index a=0 satisfies a < n-1.
    # Thus, if k=1, the condition is always met due to the problem constraints.
    if k == 1:
        return True

    # --- Prefix Sum Approach (O(n) time, O(n) space) ---

    # 1. Create the 'is_inc' array.
    # is_inc[i] = 1 if nums[i] < nums[i+1], else 0.
    # This array indicates if there's a strict increase between nums[i] and nums[i+1].
    # The length of is_inc is n-1 (indices 0 to n-2).
    is_inc = [0] * (n - 1)
    for i in range(n - 1):
        if nums[i] < nums[i + 1]:
            is_inc[i] = 1

    # 2. Calculate prefix sums of the 'is_inc' array.
    # ps[i] will store the sum of is_inc[0] + is_inc[1] + ... + is_inc[i-1].
    # The length of ps is n. ps[0] is initialized to 0.
    m = n - 1  # length of is_inc
    ps = [0] * n # ps has length n (indices 0 to n-1)
    for i in range(m): # i goes from 0 to n-2
        ps[i + 1] = ps[i] + is_inc[i]

    # 3. Define a helper function to check if a subarray is strictly increasing.
    # (Alternatively, this logic can be inlined in the main loop).
    def check_increasing(start_index: int) -> bool:
        """Checks if the subarray nums[start_index : start_index + k] is strictly increasing."""
        # A subarray nums[start:start+k] is strictly increasing if and only if
        # nums[j] < nums[j+1] for all j from start to start + k - 2.
        # This requires exactly k-1 successful comparisons.
        # In terms of the 'is_inc' array, this means the sum of 
        # is_inc[start] + is_inc[start+1] + ... + is_inc[start+k-2] must be equal to k-1.
        
        # We can compute this sum efficiently using the prefix sum array 'ps':
        # Sum = ps[ (start + k - 2) + 1 ] - ps[start]
        # Sum = ps[start + k - 1] - ps[start]
        
        required_sum = k - 1
        
        # Calculate the actual sum using prefix sums.
        # The indices start and start + k - 1 are guaranteed to be within the bounds of ps (0 to n-1)
        # because of the range of the main loop iterating 'a'.
        actual_sum = ps[start_index + k - 1] - ps[start_index]
        
        return actual_sum == required_sum

    # 4. Iterate through possible starting indices 'a' for the first subarray.
    # The first subarray is nums[a : a+k].
    # The second (adjacent) subarray is nums[a+k : a+2*k].
    # The second subarray must fit within the bounds of nums. Its last element is at index a + 2*k - 1.
    # So, we need a + 2*k - 1 < n, which implies a < n - 2*k + 1, or a <= n - 2*k.
    # The loop iterates 'a' from 0 up to and including n - 2*k.
    for a in range(n - 2 * k + 1):
        # Check if the first subarray starting at 'a' is strictly increasing.
        if check_increasing(a):
            # If the first is increasing, check the adjacent subarray starting at 'a+k'.
            if check_increasing(a + k):
                # Found two adjacent, strictly increasing subarrays of length k.
                return True
    
    # If the loop completes without finding such a pair, they do not exist.
    return False