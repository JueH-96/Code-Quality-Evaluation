import math
from collections import defaultdict
from typing import List

class Solution:
  """
  Finds the minimum number of seconds needed to make all elements in the array nums equal.
  """
  def minimumSeconds(self, nums: List[int]) -> int:
    """
    Calculates the minimum seconds based on the maximum distance between occurrences of each number.

    The core idea is that to make all elements equal to a value `v`, every index `i` must be reachable 
    from an initial position of `v` within `k` seconds. This propagation happens circularly.
    In `k` seconds, a value at index `p` can reach any index `j` such that the circular distance 
    `dist(p, j) <= k`.
    For all elements to become `v` in `k` seconds, every index `i` must satisfy `min_{p} dist(i, p) <= k`, 
    where `p` iterates over all initial indices of `v`.
    This condition is equivalent to `max_{i} min_{p} dist(i, p) <= k`.
    The maximum of the minimum distances occurs at the midpoints between consecutive occurrences of `v`.
    If the largest distance between consecutive occurrences of `v` (circularly) is `max_dist`, 
    then the furthest any index can be from an occurrence of `v` is `floor(max_dist / 2)`.
    Therefore, the minimum time `k` required for a target value `v` is `k_v = floor(max_dist / 2)`.
    The overall minimum time is the minimum of `k_v` over all possible target values `v` present in `nums`.

    Args:
      nums: A list of integers representing the initial array.

    Returns:
      The minimum number of seconds required to make all elements equal.
    """
    n = len(nums)
    
    # Constraints state 1 <= n. If n=1, the array is already equal, requiring 0 seconds.
    if n == 1:
        return 0

    # Store indices for each unique number
    # Using defaultdict simplifies initialization: `pos[x]` automatically becomes an empty list if `x` is new.
    pos = defaultdict(list)
    for i, x in enumerate(nums):
      pos[x].append(i) # Indices are added in increasing order

    # Initialize minimum seconds required. 
    # n is a safe upper bound (actually n // 2 is the tightest upper bound).
    min_k = n 

    # Iterate through each unique number present in the array. Each unique number is a potential target value.
    for x in pos:
      indices = pos[x] # Get the list of indices where x appears
      m = len(indices) # Number of occurrences of x
      
      # Calculate the maximum distance between consecutive occurrences of x in the circular array.
      if m == 1:
        # If the number appears only once, the gap is the entire circle length n.
        # The value must propagate across the whole array from this single point.
        # The furthest points are approximately n/2 distance away. The maximum gap is effectively n.
        max_dist = n
      else:
        # Calculate maximum distance between adjacent occurrences (non-wrap-around)
        max_dist = 0
        for i in range(m - 1):
          # Distance between index i+1 and index i
          dist = indices[i+1] - indices[i]
          max_dist = max(max_dist, dist)
        
        # Calculate the wrap-around distance between the last and first occurrence.
        # This is the distance from indices[m-1] forwards to indices[0] in the circular array.
        # Example: n=10, indices=[1, 8]. Last is 8, First is 1. Wrap distance is (10-8) + 1 = 3.
        # Formula: n - indices[m-1] + indices[0]
        wrap_around_dist = n - indices[m-1] + indices[0]
        max_dist = max(max_dist, wrap_around_dist)

      # The minimum time required to make all elements equal to x is determined by the largest gap (`max_dist`).
      # In k seconds, a value can spread k steps in both directions (clockwise and counter-clockwise).
      # To cover the largest gap of size `max_dist` between two occurrences, the signals propagating from
      # both ends meet roughly in the middle. The time needed for the value to reach the furthest 
      # point in this gap is floor(max_dist / 2).
      k_x = max_dist // 2 # Use integer division for floor operation
      
      # Keep track of the minimum time found so far across all unique numbers.
      min_k = min(min_k, k_x)
            
    # Return the overall minimum time required.
    return min_k