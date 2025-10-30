from typing import List

class Solution:
    def maximumWeight(self, intervals: List[List[int]]) -> List[int]:
        n = len(intervals)
        memo = {}

        def is_non_overlapping(current_interval_index, chosen_indices):
            current_interval = intervals[current_interval_index]
            for chosen_index in chosen_indices:
                chosen_interval = intervals[chosen_index]
                if not (current_interval[1] < chosen_interval[0] or chosen_interval[1] < current_interval[0]):
                    return False
            return True

        def calculate_weight(chosen_indices):
            weight = 0
            for index in chosen_indices:
                weight += intervals[index][2]
            return weight

        def solve(index, chosen_indices):
            if len(chosen_indices) == 4 or index == n:
                return calculate_weight(chosen_indices), tuple(chosen_indices)
            
            memo_key = (index, tuple(sorted(chosen_indices)))
            if memo_key in memo:
                return memo[memo_key]

            # Option 1: Skip current interval
            weight1, indices1 = solve(index + 1, chosen_indices)
            
            # Option 2: Choose current interval if possible
            weight2 = -1
            indices2 = None
            if is_non_overlapping(index, chosen_indices):
                weight2, indices2 = solve(index + 1, chosen_indices + [index])
            
            if weight2 > weight1:
                result = (weight2, indices2)
            elif weight2 < weight1:
                result = (weight1, indices1)
            else: # weight2 == weight1
                if indices2 is None:
                    result = (weight1, indices1)
                elif indices1 is None:
                    result = (weight2, indices2)
                elif indices2 < indices1:
                    result = (weight2, indices2)
                else:
                    result = (weight1, indices1)
                    
            memo[memo_key] = result
            return result

        max_weight, best_indices_tuple = solve(0, [])
        return list(best_indices_tuple)