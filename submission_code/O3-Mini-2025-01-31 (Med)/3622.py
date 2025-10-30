from bisect import bisect_left, bisect_right
from typing import List

class Solution:
    def maxFrequency(self, nums: List[int], k: int, numOperations: int) -> int:
        # Sort the array.
        nums.sort()
        n = len(nums)
        
        # We'll use a dictionary to record the frequency of each distinct number
        # Alternatively, we can iterate using two pointers over the sorted array.
        # But here we choose to iterate over the set of distinct numbers.
        # For each candidate T (which we choose as a number appearing in nums),
        # the indices eligible to be T (via operation if needed) are those numbers in
        # the interval [T - k, T + k]. Let eligible = count of nums in [T - k, T + k].
        # And count_equal = frequency of T in nums.
        # Then, if we can only fix at most numOperations indices (that are not already T),
        # the maximum frequency for T is:
        #    count_equal + min(numOperations, eligible - count_equal)
        # which is equivalently: min(count_equal + numOperations, eligible).
        
        # First, build frequency count for each distinct number.
        freq = {}
        for x in nums:
            freq[x] = freq.get(x, 0) + 1
        
        ans = 0
        # Iterate over each candidate T in sorted order (distinct ones)
        # Use binary search to find count of numbers in [T-k, T+k]
        for T in freq:
            count_equal = freq[T]
            left_bound = T - k
            right_bound = T + k
            left_idx = bisect_left(nums, left_bound)
            right_idx = bisect_right(nums, right_bound)
            eligible = right_idx - left_idx
            
            # Maximum frequency we can achieve by converting at most numOperations of
            # the eligible numbers that are not originally T.
            candidate = count_equal + min(numOperations, eligible - count_equal)
            if candidate > ans:
                ans = candidate
                
        return ans