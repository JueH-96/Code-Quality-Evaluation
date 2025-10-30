import collections
from typing import List

class Solution:
  """
  Checks if an array nums is a "good" array.
  A "good" array is a permutation of base[n] = [1, 2, ..., n-1, n, n].
  """
  def isGood(self, nums: List[int]) -> bool:
    """
    Determines if the given integer array nums is a "good" array.
    An array is "good" if it is a permutation of base[n] = [1, 2, ..., n-1, n, n]
    for some integer n >= 1.

    Args:
      nums: The input list of integers. 
            Constraints: 1 <= nums.length <= 100, 1 <= nums[i] <= 200.

    Returns:
      True if nums is a good array, False otherwise.
    """
    
    # Determine the potential value of n based on the length of nums.
    # If nums is a permutation of base[n], its length must be n + 1.
    # Therefore, n = len(nums) - 1.
    n = len(nums) - 1
    
    # The definition of base[n] requires an array of length n + 1 containing
    # numbers 1 to n-1 exactly once, and n exactly twice.
    # This definition implies n must be at least 1.
    # For n = 1, base[1] = [1, 1], length is 2.
    # If n = 0, then len(nums) = 1. Let nums = [k]. 
    # The constraints state 1 <= k <= 200. So k >= 1.
    # For an array to be good, its maximum element must be equal to n.
    # Here, max element is k >= 1, while n = 0. Since k != n, an array of length 1 cannot be good.
    # Thus, we require n >= 1, which means len(nums) must be at least 2.
    if n <= 0:
        return False

    # Use collections.Counter to count the frequency of each number in nums.
    # This takes O(L) time, where L = len(nums).
    # The space complexity is O(k), where k is the number of unique elements.
    # In the case of a good array, k = n, so space is O(n) = O(L).
    counts = collections.Counter(nums)
    
    # For nums to be a permutation of base[n], it must contain exactly n distinct numbers: {1, 2, ..., n}.
    # Check if the number of unique keys in the Counter is exactly n.
    # If len(counts) != n, it means either:
    # - Some numbers in the range [1, n] are missing.
    # - There are numbers outside the range [1, n] (e.g., > n). 
    #   (Numbers < 1 are impossible due to constraints 1 <= nums[i]).
    # In either case, nums cannot be a permutation of base[n].
    if len(counts) != n:
        return False
        
    # Check if numbers from 1 up to n-1 appear exactly once.
    # The loop iterates from i = 1 to n-1.
    # If n = 1, range(1, 1) is empty, and this loop does not execute, which is correct for base[1] = [1, 1].
    # Each lookup counts[i] takes O(1) time on average. The loop runs n-1 times. Total O(n).
    for i in range(1, n):
        # If counts[i] is not 1, it means either i is missing (counts[i] == 0)
        # or i appears more than once (counts[i] > 1). In either case, nums is not good.
        # Note: `counts[i]` will return 0 if `i` is not a key in the counter.
        if counts[i] != 1:
            return False
            
    # Check if the number n (which must be the maximum element in a good array) appears exactly twice.
    # This takes O(1) time on average.
    if counts[n] != 2:
        return False
            
    # If all the above conditions are met:
    # 1. The length of nums is n + 1 (derived from n = len(nums) - 1).
    # 2. The array contains exactly n distinct elements.
    # 3. The elements are {1, 2, ..., n}. (Implied by len(counts) == n and checks below).
    # 4. Elements 1 to n-1 appear exactly once.
    # 5. Element n appears exactly twice.
    # These conditions perfectly match the definition of base[n].
    # Therefore, nums is a permutation of base[n], and it is a good array.
    return True