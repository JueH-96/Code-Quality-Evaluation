class Solution:
    def maximumWeight(self, intervals: List[List[int]]) -> List[int]:
        n = len(intervals)

        max_weight = 0
        best_indices = []

        def are_non_overlapping(int1, int2):
            return int1[1] < int2[0] or int2[1] < int1[0]

        def update_best_indices(current_weight, current_indices):
            nonlocal max_weight, best_indices
            sorted_current_indices = sorted(current_indices)
            if current_weight > max_weight:
                max_weight = current_weight
                best_indices = sorted_current_indices
            elif current_weight == max_weight and best_indices:
                if sorted_current_indices < best_indices:
                    best_indices = sorted_current_indices
            elif current_weight == max_weight and not best_indices:
                best_indices = sorted_current_indices

        # Case 1: 1 interval
        for i in range(n):
            update_best_indices(intervals[i][2], [i])

        # Case 2: 2 intervals
        for i in range(n):
            for j in range(i + 1, n):
                if are_non_overlapping(intervals[i], intervals[j]):
                    update_best_indices(intervals[i][2] + intervals[j][2], [i, j])

        # Case 3: 3 intervals
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    if (are_non_overlapping(intervals[i], intervals[j]) and
                        are_non_overlapping(intervals[i], intervals[k]) and
                        are_non_overlapping(intervals[j], intervals[k])):
                        update_best_indices(intervals[i][2] + intervals[j][2] + intervals[k][2], [i, j, k])

        # Case 4: 4 intervals
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    for l in range(k + 1, n):
                        if (are_non_overlapping(intervals[i], intervals[j]) and
                            are_non_overlapping(intervals[i], intervals[k]) and
                            are_non_overlapping(intervals[i], intervals[l]) and
                            are_non_overlapping(intervals[j], intervals[k]) and
                            are_non_overlapping(intervals[j], intervals[l]) and
                            are_non_overlapping(intervals[k], intervals[l])):
                            update_best_indices(intervals[i][2] + intervals[j][2] + intervals[k][2] + intervals[l][2], [i, j, k, l])

        return best_indices