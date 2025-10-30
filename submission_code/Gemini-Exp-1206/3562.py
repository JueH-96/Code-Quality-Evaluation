class Solution:
    def maximumWeight(self, intervals: List[List[int]]) -> List[int]:
        n = len(intervals)
        intervals_with_indices = []
        for i in range(n):
            intervals_with_indices.append((intervals[i][0], intervals[i][1], intervals[i][2], i))
        intervals_with_indices.sort()

        best_score = 0
        best_indices = []

        def is_overlapping(interval1, interval2):
            return not (interval1[1] < interval2[0] or interval2[1] < interval1[0])

        def find_best(current_indices, current_score):
            nonlocal best_score, best_indices

            if len(current_indices) > 4:
                return

            if current_score > best_score:
                best_score = current_score
                best_indices = current_indices[:]
            elif current_score == best_score and current_score > 0:
                temp_indices = [intervals_with_indices[i][3] for i in current_indices]
                temp_indices.sort()
                
                best_temp_indices = [intervals_with_indices[i][3] for i in best_indices]
                best_temp_indices.sort()
                
                if temp_indices < best_temp_indices:
                    best_indices = current_indices[:]

            if len(current_indices) == 4:
                return

            start_index = 0 if not current_indices else current_indices[-1] + 1

            for i in range(start_index, n):
                overlapping = False
                for j in current_indices:
                    if is_overlapping(intervals_with_indices[i], intervals_with_indices[j]):
                        overlapping = True
                        break
                if not overlapping:
                    find_best(current_indices + [i], current_score + intervals_with_indices[i][2])

        find_best([], 0)
        
        result = [intervals_with_indices[i][3] for i in best_indices]
        result.sort()
        return result