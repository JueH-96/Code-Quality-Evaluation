class Solution:
    def maximumWeight(self, intervals: List[List[int]]) -> List[int]:
        n = len(intervals)
        intervals_with_index = []
        for i in range(n):
            intervals_with_index.append((intervals[i][0], intervals[i][1], intervals[i][2], i + 1))
        
        intervals_with_index.sort()
        
        dp = {}
        
        def solve(index, count):
            if count == 0:
                return 0, []
            
            if index == n:
                return 0, []
            
            if (index, count) in dp:
                return dp[(index, count)]
            
            # Option 1: Don't include the current interval
            score1, indices1 = solve(index + 1, count)
            
            # Option 2: Include the current interval
            current_start, current_end, current_weight, current_index = intervals_with_index[index]
            
            next_index = index + 1
            while next_index < n and intervals_with_index[next_index][0] <= current_end:
                next_index += 1
            
            score2, indices2 = solve(next_index, count - 1)
            score2 += current_weight
            indices2 = [current_index] + indices2
            
            if score2 > score1:
                dp[(index, count)] = score2, indices2
                return score2, indices2
            elif score2 == score1:
                if indices2 < indices1:
                    dp[(index, count)] = score2, indices2
                    return score2, indices2
                else:
                    dp[(index, count)] = score1, indices1
                    return score1, indices1
            else:
                dp[(index, count)] = score1, indices1
                return score1, indices1
        
        best_score = 0
        best_indices = []
        
        for count in range(1, min(5, n + 1)):
            score, indices = solve(0, count)
            if score > best_score:
                best_score = score
                best_indices = indices
            elif score == best_score:
                if indices < best_indices or not best_indices:
                    best_indices = indices
        
        return best_indices