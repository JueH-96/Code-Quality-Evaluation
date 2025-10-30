from typing import List

class Solution:
    def countWays(self, nums: List[int]) -> int:
        # Sort the thresholds to be able to use prefix-information
        nums.sort()
        n = len(nums)
        ways = 0
        
        # Try every possible group size k from 0 to n
        # (k = number of selected students)
        for k in range(n + 1):
            # All elements strictly smaller than k are located before index k
            left_ok  = (k == 0) or (nums[k - 1] < k)   # biggest of the first k elements < k
            # All elements from index k onward are >= nums[k]
            right_ok = (k == n) or (nums[k] > k)       # smallest of the rest > k
            if left_ok and right_ok:
                ways += 1
        
        return ways