class Solution:
    def maximumWeight(self, intervals: List[List[int]]) -> List[int]:
        from sortedcontainers import SortedList
        import bisect
        
        # Sort intervals by their end time, then start time, then weight (descending)
        intervals = sorted((x[1], x[0], x[2], i) for i, x in enumerate(intervals))
        
        # dp[i] will store the maximum weight we can achieve using the first i intervals
        # and the indices of the intervals used to achieve this weight
        dp = [(0, [])]  # initial state: no intervals, no weight
        
        # To efficiently find the rightmost non-overlapping interval
        ends = SortedList()  # to keep track of end times of intervals
        
        for end, start, weight, index in intervals:
            # Find the largest interval that ends before 'start' to ensure non-overlap
            pos = ends.bisect_right((start, float('inf'))) - 1
            
            # If there's a valid interval that doesn't overlap
            if pos >= 0:
                max_weight, indices = dp[pos]
                new_weight = max_weight + weight
            else:
                new_weight = weight
                indices = []
            
            # Only update dp if we found a better option
            if new_weight > dp[-1][0]:
                dp.append((new_weight, indices + [index]))
            else:
                dp.append(dp[-1])
            
            ends.add((end, len(dp) - 1))
        
        # Get the indices of the chosen intervals from the last element of dp
        max_indices = dp[-1][1]
        
        # Sort indices to return the lexicographically smallest array
        return sorted(max_indices)