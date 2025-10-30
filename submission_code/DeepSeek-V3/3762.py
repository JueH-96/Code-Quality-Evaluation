class Solution:
    def maxScore(self, points: List[int], m: int) -> int:
        left = 1
        right = sum(points)
        answer = 0
        
        def is_possible(x):
            total_moves = 0
            current_sum = 0
            n = len(points)
            i = 0
            while i < n:
                if points[i] >= x:
                    total_moves += 1
                    i += 1
                else:
                    current_sum = points[i]
                    j = i + 1
                    while j < n and current_sum + points[j] < x:
                        current_sum += points[j]
                        j += 1
                    if j >= n:
                        return False
                    segment_length = j - i + 1
                    total_moves += 2 * (segment_length - 1)
                    i = j + 1
                    if total_moves > m:
                        return False
            return total_moves <= m
        
        while left <= right:
            mid = (left + right) // 2
            if is_possible(mid):
                answer = mid
                left = mid + 1
            else:
                right = mid - 1
        return answer