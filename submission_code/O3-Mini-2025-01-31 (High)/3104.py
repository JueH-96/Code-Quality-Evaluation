from typing import List
import bisect

class Solution:
    def countWays(self, nums: List[int]) -> int:
        # First sort the array.
        nums.sort()
        n = len(nums)
        ways = 0
        
        # We try every possible m from 0 (empty selection) to n (all selected)
        # Under the conditions:
        #   • A student with nums[i] < m must be selected.
        #   • A student with nums[i] > m must not be selected.
        #   • A student with nums[i] == m can never be happy.
        # Hence we require that:
        #   1) The number of students with nums[i] < m is exactly m.
        #   2) There are no students with nums[i] equal to m.
        for m in range(n + 1):
            # count of students with a threshold strictly less than m.
            count_less = bisect.bisect_left(nums, m)
            # count of students with threshold exactly equal to m.
            count_eq = bisect.bisect_right(nums, m) - count_less
            
            if count_less == m and count_eq == 0:
                ways += 1
        
        return ways