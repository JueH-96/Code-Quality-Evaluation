import bisect
from typing import List

class Solution:
    def countWays(self, nums: List[int]) -> int:
        n = len(nums)
        
        # Sort the array nums. This allows for efficient counting of elements
        # less than, equal to, or greater than a specific value using bisect.
        nums.sort()
        
        total_ways = 0
        
        # Iterate through all possible numbers of selected students, k,
        # from 0 to n (inclusive).
        for k in range(n + 1):
            # 1. Check if any student has nums[i] == k.
            #    bisect_left(nums, k) finds the insertion point for k to keep the list sorted.
            #    All elements before this point are < k.
            #    bisect_right(nums, k) finds the insertion point for k such that all elements
            #    before this point are <= k.
            #    The difference between bisect_right and bisect_left gives the count of elements equal to k.
            
            idx_less_than_k = bisect.bisect_left(nums, k)
            idx_greater_than_or_equal_to_k = bisect.bisect_right(nums, k)
            
            # If idx_greater_than_or_equal_to_k > idx_less_than_k, it means k is present in nums.
            # If k is present, there is a student with nums[i] == k, who cannot be happy.
            if idx_greater_than_or_equal_to_k > idx_less_than_k:
                continue # This 'k' is not a valid number of selected students.
            
            # 2. Check if the number of students who *must* be selected (those with nums[i] < k)
            #    is exactly equal to the current 'k'.
            #    'idx_less_than_k' already gives us this count.
            if idx_less_than_k == k:
                # If both conditions are met, 'k' is a valid way to select students.
                # All students with nums[i] < k (exactly 'k' of them) are selected and happy.
                # All students with nums[i] > k (the remaining 'n - k' students, since no nums[i] == k)
                # are not selected and happy.
                total_ways += 1
                
        return total_ways