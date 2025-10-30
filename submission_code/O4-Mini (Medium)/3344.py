from typing import List

class Solution:
    def minimumDistance(self, points: List[List[int]]) -> int:
        n = len(points)
        # Compute transformed coordinates u = x+y, v = x-y
        u = [x + y for x, y in points]
        v = [x - y for x, y in points]

        # Find max1, max2, count of max1 for u
        max_u1 = -10**20
        max_u2 = -10**20
        cnt_max_u1 = 0
        # Find min1, min2, count of min1 for u
        min_u1 = 10**20
        min_u2 = 10**20
        cnt_min_u1 = 0

        # Similarly for v
        max_v1 = -10**20
        max_v2 = -10**20
        cnt_max_v1 = 0
        min_v1 = 10**20
        min_v2 = 10**20
        cnt_min_v1 = 0

        for val in u:
            if val > max_u1:
                max_u2 = max_u1
                max_u1 = val
                cnt_max_u1 = 1
            elif val == max_u1:
                cnt_max_u1 += 1
            elif val > max_u2:
                max_u2 = val

            if val < min_u1:
                min_u2 = min_u1
                min_u1 = val
                cnt_min_u1 = 1
            elif val == min_u1:
                cnt_min_u1 += 1
            elif val < min_u2:
                min_u2 = val

        for val in v:
            if val > max_v1:
                max_v2 = max_v1
                max_v1 = val
                cnt_max_v1 = 1
            elif val == max_v1:
                cnt_max_v1 += 1
            elif val > max_v2:
                max_v2 = val

            if val < min_v1:
                min_v2 = min_v1
                min_v1 = val
                cnt_min_v1 = 1
            elif val == min_v1:
                cnt_min_v1 += 1
            elif val < min_v2:
                min_v2 = val

        # Collect candidate indices: those achieving any of the four extremes
        candidates = set()
        for i in range(n):
            if u[i] == max_u1 or u[i] == min_u1 or v[i] == max_v1 or v[i] == min_v1:
                candidates.add(i)

        ans = 10**30
        # Try removing each candidate
        for i in candidates:
            # new max_u
            if u[i] == max_u1 and cnt_max_u1 == 1:
                new_max_u = max_u2
            else:
                new_max_u = max_u1
            # new min_u
            if u[i] == min_u1 and cnt_min_u1 == 1:
                new_min_u = min_u2
            else:
                new_min_u = min_u1
            # new max_v
            if v[i] == max_v1 and cnt_max_v1 == 1:
                new_max_v = max_v2
            else:
                new_max_v = max_v1
            # new min_v
            if v[i] == min_v1 and cnt_min_v1 == 1:
                new_min_v = min_v2
            else:
                new_min_v = min_v1

            # the diameter after removal
            dist = max(new_max_u - new_min_u, new_max_v - new_min_v)
            ans = min(ans, dist)

        return ans