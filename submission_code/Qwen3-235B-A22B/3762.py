from typing import List

class Solution:
    def maxScore(self, points: List[int], m: int) -> int:
        def is_possible(k):
            sum_rv = 0
            max_rv = 0
            for p in points:
                if k == 0:
                    req = 0
                else:
                    req = (k + p - 1) // p
                sum_rv += req
                if req > max_rv:
                    max_rv = req
            if sum_rv > m:
                return False
            # Check if the maximal required visits can be done within m moves
            if 2 * max_rv - 1 > m:
                return False
            return True
        
        left = 0
        # Find an appropriate right bound for binary search
        right = 0
        for p in points:
            current_upper = p * m
            if current_upper > right:
                right = current_upper
        # Initial doubling to find a suitable right if needed
        while not is_possible(right):
            right //= 2
            if right == 0:
                break
        # Binary search to find the maximum feasible k
        while left < right:
            mid = (left + right + 1) // 2
            if is_possible(mid):
                left = mid
            else:
                right = mid - 1
        return left