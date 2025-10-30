from typing import List

class Solution:
    def maxPossibleScore(self, start: List[int], d: int) -> int:
        n = len(start)
        def is_possible(score):
            indices = sorted(range(n), key=lambda i: start[i])
            chosen_values = []
            current_val = start[indices[0]]
            chosen_values.append(current_val)
            for i in range(1, n):
                current_val = max(start[indices[i]], chosen_values[-1] + score)
                if current_val > start[indices[i]] + d:
                    return False
                chosen_values.append(current_val)
            return True

        low = 0
        high = 2 * 10**9 + 1
        ans = 0
        while low <= high:
            mid = (low + high) // 2
            if is_possible(mid):
                ans = mid
                low = mid + 1
            else:
                high = mid - 1
        return ans