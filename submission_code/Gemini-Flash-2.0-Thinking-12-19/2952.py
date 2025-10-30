from typing import List

class Solution:
    def minimumTime(self, nums1: List[int], nums2: List[int], x: int) -> int:
        n = len(nums1)
        memo = {}
        
        def get_sum(current_nums1):
            return sum(current_nums1)
            
        def solve(time_remaining, current_nums1_tuple):
            current_nums1 = list(current_nums1_tuple)
            if get_sum(current_nums1) <= x:
                return True
            if time_remaining == 0:
                return False
            if (time_remaining, current_nums1_tuple) in memo:
                return memo[(time_remaining, current_nums1_tuple)]
            
            possible = False
            for i in range(n):
                next_nums1 = list(current_nums1)
                for j in range(n):
                    next_nums1[j] += nums2[j]
                next_nums1[i] = 0
                if solve(time_remaining - 1, tuple(next_nums1)):
                    possible = True
                    break
                    
            memo[(time_remaining, current_nums1_tuple)] = possible
            return possible
            
        for t in range(n * 1001): # Iterate through time from 0 up to a reasonable limit
            memo = {} # Clear memoization for each time value
            if solve(t, tuple(nums1)):
                return t
                
        return -1