class Solution:
    def maximumWeight(self, intervals: List[List[int]]) -> List[int]:
        n = len(intervals)
        
        # Add original indices and sort by end time
        indexed_intervals = [(intervals[i][0], intervals[i][1], intervals[i][2], i) for i in range(n)]
        indexed_intervals.sort(key=lambda x: x[1])
        
        # For each interval, find the latest non-overlapping interval before it
        def find_prev(i):
            for j in range(i - 1, -1, -1):
                if indexed_intervals[j][1] < indexed_intervals[i][0]:
                    return j
            return -1
        
        # DP with memoization
        memo = {}
        
        def dp(i, count):
            if count == 0:
                return (0, [])
            if i < 0:
                return (-1, [])
            
            if (i, count) in memo:
                return memo[(i, count)]
            
            # Option 1: don't take interval i
            weight1, indices1 = dp(i - 1, count)
            
            # Option 2: take interval i
            prev_i = find_prev(i)
            if prev_i >= 0:
                weight2, indices2 = dp(prev_i, count - 1)
                if weight2 >= 0:
                    weight2 += indexed_intervals[i][2]
                    indices2 = indices2 + [indexed_intervals[i][3]]
                else:
                    weight2 = -1
                    indices2 = []
            elif count == 1:
                weight2 = indexed_intervals[i][2]
                indices2 = [indexed_intervals[i][3]]
            else:
                weight2 = -1
                indices2 = []
            
            # Choose the better option
            if weight2 > weight1:
                result = (weight2, indices2)
            elif weight2 == weight1 and weight2 >= 0:
                # Choose lexicographically smaller
                if sorted(indices2) < sorted(indices1):
                    result = (weight2, indices2)
                else:
                    result = (weight1, indices1)
            else:
                result = (weight1, indices1)
            
            memo[(i, count)] = result
            return result
        
        best_weight = -1
        best_indices = []
        
        # Try all possible counts from 1 to 4
        for count in range(1, 5):
            weight, indices = dp(n - 1, count)
            if weight > best_weight:
                best_weight = weight
                best_indices = indices
            elif weight == best_weight and weight >= 0:
                if sorted(indices) < sorted(best_indices):
                    best_indices = indices
        
        return sorted(best_indices)