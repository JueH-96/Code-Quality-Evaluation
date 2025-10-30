from typing import List

class Solution:
  def minDifference(self, nums: List[int]) -> int:
    
    def check(d: int) -> bool:
      n = len(nums)
      # A safe large value for the upper bound.
      # max(nums) + n * max(d) ~ 10^9 + 10^5 * 10^9, which can be large.
      large_val = 2 * 10**14 
      
      low = [0] * n
      high = [0] * n
      
      has_missing = False
      for i in range(n):
        if nums[i] == -1:
          low[i] = 1
          high[i] = large_val
          has_missing = True
        else:
          low[i] = nums[i]
          high[i] = nums[i]

      # Propagate constraints from left to right.
      for i in range(1, n):
        low[i] = max(low[i], low[i-1] - d)
        high[i] = min(high[i], high[i-1] + d)
        
      # Propagate constraints from right to left.
      for i in range(n - 2, -1, -1):
        low[i] = max(low[i], low[i+1] - d)
        high[i] = min(high[i], high[i+1] + d)

      # After propagation, if any range is invalid, d is not possible.
      for i in range(n):
        if low[i] > high[i]:
          return False

      if not has_missing:
        return True

      # Find the intersection of all ranges for '-1' positions.
      min_replace_val = 0
      max_replace_val = large_val + 1
      for i in range(n):
        if nums[i] == -1:
          min_replace_val = max(min_replace_val, low[i])
          max_replace_val = min(max_replace_val, high[i])

      # Case 1: The intersection is non-empty. We can pick a single value (x=y).
      if min_replace_val <= max_replace_val:
        return True
      
      # Case 2: Intersection is empty. We need two values x and y.
      # It can be shown that if a solution exists, one can be found by picking
      # x and y at the boundaries of the union of ranges.
      # The condition becomes |x-y| <= d.
      if min_replace_val - max_replace_val <= d:
        return True
        
      return False

    # Binary search for the minimum possible difference `d`.
    # The difference can be between 0 and 10^9.
    left, right = 0, 10**9
    ans = right
    
    while left <= right:
      mid = (left + right) // 2
      if check(mid):
        ans = mid
        right = mid - 1
      else:
        left = mid + 1
        
    return ans