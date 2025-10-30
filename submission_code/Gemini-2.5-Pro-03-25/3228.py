import collections
from typing import List

class Solution:
  """
  Solves the maximum set size problem.
  Finds the maximum possible size of a set s formed by the remaining elements 
  after removing n / 2 elements from nums1 and n / 2 elements from nums2.
  """
  def maximumSetSize(self, nums1: List[int], nums2: List[int]) -> int:
    """
    Calculates the maximum possible size of the set s.

    Args:
      nums1: The first list of integers.
      nums2: The second list of integers. Both lists have the same even length n.

    Returns:
      The maximum possible size of the set s.
    """
    
    n = len(nums1)
    # n is guaranteed to be even, so n // 2 performs integer division which is equivalent to n / 2 for even n.
    half_n = n // 2
    
    # Compute the set of unique elements in nums1 and nums2 using Python's set data structure.
    # This automatically handles duplicates and provides efficient membership testing and set operations.
    U1 = set(nums1)
    U2 = set(nums2)
    
    # Compute the sizes of the sets of unique elements.
    size_U1 = len(U1)
    size_U2 = len(U2)
    
    # Compute the union of the unique elements from both lists.
    # The union represents the total pool of distinct numbers available across both initial lists.
    # The size of the union gives an upper bound on the size of the final set s.
    U1_union_U2 = U1 | U2 # Using the '|' operator for set union. Equivalent to U1.union(U2)
    size_U1_union_U2 = len(U1_union_U2)
    
    # Calculate the maximum number of unique elements that could potentially remain in nums1
    # after removing half_n elements. Let this potential maximum count be N1.
    # After removals, n/2 elements remain in nums1. The number of unique elements among these
    # remaining elements cannot exceed n/2.
    # Additionally, the number of unique elements cannot exceed the total count of unique elements 
    # initially present in nums1 (size_U1).
    # Therefore, the maximum possible number of unique elements from nums1 in the final set is min(size_U1, half_n).
    N1 = min(size_U1, half_n)
    
    # Similarly, calculate the maximum number of unique elements N2 that could potentially remain from nums2.
    N2 = min(size_U2, half_n)
    
    # Let S1 be the set of unique elements chosen from the remaining elements of nums1, 
    # and S2 be the set of unique elements chosen from the remaining elements of nums2.
    # The final set 's' is the union of S1 and S2: s = S1 U S2.
    # We want to maximize the size of s, which is |s| = |S1 U S2|.
    
    # The size of the final set 's' is constrained by two main factors:
    # 1. The total number of distinct elements available across both lists: |U1 U U2| = size_U1_union_U2.
    #    The final set 's' must be a subset of U1 U U2.
    # 2. The sum of the maximum possible number of unique elements contributed from each list: N1 + N2.
    #    This follows from the set property |S1 U S2| <= |S1| + |S2|. Since |S1| <= N1 and |S2| <= N2,
    #    we have |S1 U S2| <= N1 + N2.
    
    # Combining these two upper bounds, the maximum possible size of the set s is at most
    # min(size_U1_union_U2, N1 + N2).
    # It can be shown (e.g., via a greedy construction argument) that this upper bound is always achievable.
    # Therefore, this minimum value represents the maximum possible size of the set s.
    max_set_size = min(size_U1_union_U2, N1 + N2)
    
    return max_set_size