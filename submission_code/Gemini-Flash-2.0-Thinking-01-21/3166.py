from collections import Counter
from typing import List

class Solution:
    def minGroupsForValidAssignment(self, nums: List[int]) -> int:
        counts = Counter(nums)
        value_counts = list(counts.values())
        n = len(nums)
        min_groups = n
        for g in range(1, n + 1):
            s1 = n // g + 1
            s2 = n // g
            x = n % g
            y = g - x
            if s2 == 0:
                if s1 == 0:
                    continue # Should not happen for g >= 1, n >= 1
                current_s1, current_s2 = s1, s2
                current_x, current_y = x, y
            else:
                current_s1, current_s2 = s1, s2
                current_x, current_y = x, y
                
            possible_assignments = []
            valid_g = True
            for count in value_counts:
                value_possible_pairs = []
                for k1 in range(count // current_s1 + 1):
                    if (count - k1 * current_s1) % current_s2 == 0:
                        k2 = (count - k1 * current_s1) // current_s2
                        if k2 >= 0:
                            value_possible_pairs.append((k1, k2))
                if not value_possible_pairs:
                    valid_g = False
                    break
                possible_assignments.append(value_possible_pairs)
            if not valid_g:
                continue
                
            memo = {}
            
            def solve(index, remaining_x, remaining_y):
                if index == len(possible_assignments):
                    return remaining_x == 0 and remaining_y == 0
                if (index, remaining_x, remaining_y) in memo:
                    return memo[(index, remaining_x, remaining_y)]
                
                for k1, k2 in possible_assignments[index]:
                    if remaining_x >= k1 and remaining_y >= k2:
                        if solve(index + 1, remaining_x - k1, remaining_y - k2):
                            memo[(index, remaining_x, remaining_y)] = True
                            return True
                            
                memo[(index, remaining_x, remaining_y)] = False
                return False
                
            if solve(0, current_x, current_y):
                min_groups = g
                break
                
        return min_groups