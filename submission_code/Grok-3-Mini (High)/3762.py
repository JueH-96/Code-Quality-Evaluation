from typing import List

class Solution:
    def maxScore(self, points: List[int], m: int) -> int:
        left, right = 0, m * max(points)
        res = 0
        while left <= right:
            mid = (left + right) // 2
            if self.can_achieve(points, m, mid):
                res = mid
                left = mid + 1
            else:
                right = mid - 1
        return res

    def can_achieve(self, points, m, T):
        if T == 0:
            return True
        n = len(points)
        v = [0] * n
        for i in range(n):
            v[i] = (T + points[i] - 1) // points[i]  # ceil division
        if n == 2:
            v0, v1 = v[0], v[1]
            min_k = min(2 * max(v0, v1), 2 * max(v1, v0 - 1) + 1 if v0 > 0 else 0)
            return min_k <= m
        else:
            # For n > 2, use E_min from c_i
            prefix_max = [v[0]]
            for i in range(1, n):
                prefix_max.append(max(prefix_max[-1], v[i]))
            suffix_max = [v[n-1]]
            for i in range(n-2, -1, -1):
                suffix_max.append(max(suffix_max[-1], v[i]))
            suffix_max.reverse()  # suffix_max[i] = max v[i to n-1]
            E_min = 0
            for i in range(n-1):
                c_i = max(prefix_max[i], suffix_max[i+1])
                E_min += c_i
            k_min = E_min + 1
            if k_min > m:
                return False
            else:
                return True