from typing import List
from math import ceil
from collections import Counter

class Solution:
    def minGroupsForValidAssignment(self, nums: List[int]) -> int:
        n = len(nums)
        freq = Counter(nums)
        
        def is_feasible(g):
            s_prime = ceil(n / g)
            total_groups_needed = 0
            for f in freq.values():
                groups_for_f = ceil(f / s_prime)
                total_groups_needed += groups_for_f
                if total_groups_needed > g:
                    return False
            return True
        
        # Binary search over the number of groups
        left, right = 1, n
        answer = n
        while left <= right:
            mid = (left + right) // 2
            if is_feasible(mid):
                answer = mid
                right = mid - 1
            else:
                left = mid + 1
        return answer