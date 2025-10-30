class Solution:
    def maximumWeight(self, intervals: List[List[int]]) -> List[int]:
        intervals_with_indices = []
        for i, interval in enumerate(intervals):
            intervals_with_indices.append((interval[0], interval[1], interval[2], i))

        intervals_with_indices.sort()  # Sort by start time

        n = len(intervals_with_indices)
        dp = {}  # Memoization for dynamic programming

        def solve(index, count, last_end):
            if index == n or count == 4:
                return 0, []

            if (index, count, last_end) in dp:
                return dp[(index, count, last_end)]

            # Option 1: Don't include the current interval
            max_weight, max_indices = solve(index + 1, count, last_end)

            # Option 2: Include the current interval if it doesn't overlap
            current_start, current_end, current_weight, current_index = intervals_with_indices[index]
            if current_start > last_end:
                new_weight, new_indices = solve(index + 1, count + 1, current_end)
                new_weight += current_weight
                new_indices = [current_index] + new_indices

                if new_weight > max_weight:
                    max_weight = new_weight
                    max_indices = new_indices
                elif new_weight == max_weight and new_indices < max_indices:
                    max_indices = new_indices

            dp[(index, count, last_end)] = (max_weight, max_indices)
            return max_weight, max_indices

        max_weight, max_indices = solve(0, 0, 0)
        max_indices.sort()
        return max_indices