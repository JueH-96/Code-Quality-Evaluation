import heapq

class Solution:
    def maximumWeight(self, intervals: list[list[int]]) -> list[int]:
        n = len(intervals)
        
        # Sort intervals by end time
        indexed_intervals = sorted([(r, l, w, i) for i, (l, r, w) in enumerate(intervals)])
        
        max_score = 0
        best_indices = []

        for i in range(1 << n):
            chosen_intervals = []
            current_score = 0
            current_indices = []
            
            for j in range(n):
                if (i >> j) & 1:
                    chosen_intervals.append(indexed_intervals[j])
                    
            if len(chosen_intervals) > 4:
                continue
            
            
            valid = True
            if len(chosen_intervals) > 1:
                chosen_intervals.sort(key=lambda x: x[0])
                for k in range(len(chosen_intervals) - 1):
                    if chosen_intervals[k][0] >= chosen_intervals[k+1][1]:
                        valid = False
                        break
                    
            if not valid:
                continue
            
            for _, _, w, idx in chosen_intervals:
                current_score += w
                current_indices.append(idx)
            
            if current_score > max_score:
                max_score = current_score
                best_indices = sorted(current_indices)
            elif current_score == max_score:
                current_indices = sorted(current_indices)
                if not best_indices or current_indices < best_indices:
                    best_indices = current_indices
        
        return best_indices