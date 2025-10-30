class Solution:
    def maximumWeight(self, intervals: List[List[int]]) -> List[int]:
        n = len(intervals)
        
        def are_non_overlapping(indices):
            # Check if intervals at given indices are non-overlapping
            selected_intervals = [intervals[i] for i in indices]
            for i in range(len(selected_intervals)):
                for j in range(i + 1, len(selected_intervals)):
                    # Two intervals [a,b] and [c,d] overlap if max(a,c) <= min(b,d)
                    if max(selected_intervals[i][0], selected_intervals[j][0]) <= min(selected_intervals[i][1], selected_intervals[j][1]):
                        return False
            return True
        
        max_weight = 0
        best_indices = []
        
        # Try all combinations of 1, 2, 3, and 4 intervals
        from itertools import combinations
        
        for k in range(1, 5):  # 1 to 4 intervals
            for indices in combinations(range(n), k):
                if are_non_overlapping(indices):
                    weight = sum(intervals[i][2] for i in indices)
                    indices_list = list(indices)
                    
                    if weight > max_weight:
                        max_weight = weight
                        best_indices = indices_list
                    elif weight == max_weight:
                        # Compare lexicographically
                        if not best_indices or indices_list < best_indices:
                            best_indices = indices_list
        
        return best_indices