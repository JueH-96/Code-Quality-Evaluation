class Solution:
    def maxPossibleScore(self, start: List[int], d: int) -> int:
        n = len(start)
        intervals = sorted([(s, s + d) for s in start])
        
        def is_valid(score):
            prev = float('-inf')
            count = 0
            for left, right in intervals:
                next_valid = max(left, prev + score)
                if next_valid <= right:
                    prev = next_valid
                    count += 1
                if count == n:
                    return True
            return False
        
        left, right = 0, 10**9
        while left < right:
            mid = (left + right + 1) // 2
            if is_valid(mid):
                left = mid
            else:
                right = mid - 1
        
        return left