import collections

class Solution:
    def maximumWeight(self, intervals: List[List[int]]) -> List[int]:
        memo = {}
        n = len(intervals)

        def are_non_overlapping_with_current(current_interval_index, selected_indices):
            for selected_index in selected_indices:
                interval1 = intervals[current_interval_index]
                interval2 = intervals[selected_index]
                if not (interval1[1] < interval2[0] or interval2[1] < interval1[0]):
                    return False
            return True

        def solve(index, selected_indices_tuple):
            if (index, selected_indices_tuple) in memo:
                return memo[(index, selected_indices_tuple)]

            selected_indices = list(selected_indices_tuple)
            if len(selected_indices) == 4 or index == n:
                current_weight = sum(intervals[i][2] for i in selected_indices)
                result = (current_weight, sorted(selected_indices))
                memo[(index, selected_indices_tuple)] = result
                return result

            # Option 1: Skip interval index
            weight1, indices1 = solve(index + 1, selected_indices_tuple)

            # Option 2: Try to include interval index
            can_include = are_non_overlapping_with_current(index, selected_indices)

            if can_include:
                weight2, indices2 = solve(index + 1, tuple(sorted(selected_indices + [index])))
            else:
                weight2 = -1
                indices2 = []

            if weight2 > weight1:
                result = (weight2, indices2)
            elif weight2 < weight1:
                result = (weight1, indices1)
            else: # weight1 == weight2
                if not indices2: # weight2 failed to include
                    result = (weight1, indices1)
                elif not indices1: # weight1 failed to skip - should not happen?
                    result = (weight2, indices2)
                elif indices2 < indices1:
                    result = (weight2, indices2)
                else:
                    result = (weight1, indices1)
            memo[(index, selected_indices_tuple)] = result
            return result

        max_weight, best_indices = solve(0, tuple())
        return best_indices