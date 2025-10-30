from typing import List

class Solution:
  def maxFrequencyScore(self, nums: List[int], k: int) -> int:
    """
    Finds the maximum possible frequency of a single element in the array `nums`
    after performing at most `k` operations. An operation consists of increasing
    or decreasing an element by 1, with a cost of 1.

    The problem asks for the maximum frequency `m` we can achieve. This suggests
    that if we can achieve a frequency of `m`, we can also achieve any frequency
    less than `m`. This monotonic property makes the problem suitable for
    binary search on the answer (the frequency `m`).
    """
    n = len(nums)
    nums.sort()
    
    prefix = [0] * (n + 1)
    for i in range(n):
        prefix[i + 1] = prefix[i] + nums[i]
        
    def is_possible(m: int) -> bool:
        """
        Checks if it's possible to achieve a frequency of m with at most k cost.
        This is done by checking all contiguous subarrays of size m in the sorted array.
        """
        # A frequency of 1 is always possible with 0 cost.
        if m <= 1:
            return True
        
        # Slide a window of size m
        for i in range(n - m + 1):
            # The window is nums[i ... i+m-1]
            
            # For a window of size m, the median is one of the middle elements.
            # We choose the element at index i + m//2 as the target.
            # This choice minimizes the sum of absolute differences.
            median_idx_in_window = m // 2
            median_idx = i + median_idx_in_window
            target = nums[median_idx]
            
            # Cost for elements to the left of the median
            # These elements are nums[i...median_idx-1]. There are median_idx_in_window of them.
            # Sum of these elements: prefix[median_idx] - prefix[i]
            # Cost = (count * target) - sum
            left_cost = target * median_idx_in_window - (prefix[median_idx] - prefix[i])
            
            # Cost for elements to the right of the median (including the median itself)
            # These elements are nums[median_idx...i+m-1]. There are m - median_idx_in_window of them.
            # Sum of these elements: prefix[i+m] - prefix[median_idx]
            # Cost = sum - (count * target)
            right_cost = (prefix[i + m] - prefix[median_idx]) - target * (m - median_idx_in_window)
            
            total_cost = left_cost + right_cost
            
            if total_cost <= k:
                return True
                
        return False

    # Binary search for the maximum possible score (frequency)
    low, high = 1, n
    ans = 0 # If n=0, this would be the answer, but constraints say n>=1.
            # If n>=1, ans will be at least 1.
    while low <= high:
        mid = (low + high) // 2
        if is_possible(mid):
            ans = mid
            low = mid + 1
        else:
            high = mid - 1
            
    return ans