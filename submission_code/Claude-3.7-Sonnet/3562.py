class Solution:
    def maximumWeight(self, intervals: List[List[int]]) -> List[int]:
        n = len(intervals)
        
        # Check if two intervals overlap
        def does_overlap(i, j):
            a_start, a_end = intervals[i][0], intervals[i][1]
            b_start, b_end = intervals[j][0], intervals[j][1]
            return max(a_start, b_start) <= min(a_end, b_end)
        
        # Precompute overlaps to avoid redundant calculations
        overlaps = [[False] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                overlaps[i][j] = overlaps[j][i] = does_overlap(i, j)
        
        # Track best solution found
        max_weight = 0
        best_subset = []
        
        def backtrack(idx, curr_subset, curr_weight):
            nonlocal max_weight, best_subset
            
            # Check if current solution is better
            if curr_weight > max_weight or (curr_weight == max_weight and sorted(curr_subset) < sorted(best_subset)):
                max_weight = curr_weight
                best_subset = curr_subset.copy()
            
            # Base case: we have 4 intervals or reached the end
            if len(curr_subset) == 4 or idx == n:
                return
            
            # Skip the current interval
            backtrack(idx + 1, curr_subset, curr_weight)
            
            # Try to choose the current interval if it doesn't overlap with any chosen interval
            can_choose = True
            for j in curr_subset:
                if overlaps[idx][j]:
                    can_choose = False
                    break
            
            if can_choose:
                curr_subset.append(idx)
                backtrack(idx + 1, curr_subset, curr_weight + intervals[idx][2])
                curr_subset.pop()
        
        backtrack(0, [], 0)
        
        return sorted(best_subset)