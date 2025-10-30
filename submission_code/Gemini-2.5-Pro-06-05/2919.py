import collections
from typing import List

class Solution:
  def maxIncreasingGroups(self, usageLimits: List[int]) -> int:
    """
    Finds the maximum number of increasing groups that can be formed.

    The problem asks for the maximum number of groups `k` we can create,
    such that the groups have distinct numbers, and the size of group `i` is
    strictly greater than the size of group `i-1`. Each number `j` can be used
    at most `usageLimits[j]` times.

    To maximize `k`, we should aim for the smallest possible group sizes that
    satisfy the increasing size condition. These are sizes 1, 2, 3, ..., k.

    The total number of items (or "slots") required to form `k` groups of
    these minimal sizes is the sum 1 + 2 + ... + k, which is k * (k + 1) / 2.
    This is also known as the k-th triangular number.
    
    The core of the greedy strategy is:
    1. Sort `usageLimits` to handle the scarcest resources first.
    2. Initialize `groups = 0` (number of groups we can form) and `total_items = 0`.
    3. Iterate through the sorted `usageLimits`:
       a. Add the current limit to `total_items`.
       b. Check if the `total_items` is sufficient to form `groups + 1` groups.
          The requirement is `total_items >= (groups + 1) * (groups + 2) / 2`.
       c. If it is, we can now support one more group, so we increment `groups`.

    This works because by sorting, we ensure that when we can afford `k` groups
    in total, we have also processed at least `k` items, guaranteeing we have
    enough distinct numbers for the largest group.
    """
    usageLimits.sort()
    
    groups = 0
    total_items = 0
    
    for limit in usageLimits:
      total_items += limit
      
      # Check if we have enough items to form `groups + 1` groups.
      # Required items for groups of sizes 1, ..., groups + 1 is the (groups + 1)-th triangular number.
      required_for_next_level = (groups + 1) * (groups + 2) // 2
      
      if total_items >= required_for_next_level:
        groups += 1
        
    return groups