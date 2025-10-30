class Solution:
    def maxDistance(self, side: int, points: List[List[int]], k: int) -> int:
        n = len(points)
        L = 4 * side
        S = []
        for x, y in points:
            if x == 0:
                s = y
            elif y == side:
                s = side + x
            elif x == side:
                s = 2 * side + (side - y)
            else:
                s = 3 * side + (side - x)
            S.append(s)
        S.sort()
        T = S + [s + L for s in S]
        
        def feasible(d):
            n_points = len(S)
            total_len = 2 * n_points
            next_avail = [0] * total_len
            j = 0
            for i in range(total_len):
                if j < i:
                    j = i
                while j < total_len and T[j] - T[i] < d:
                    j += 1
                next_avail[i] = j
            
            for start in range(n_points):
                count = 1
                current = start
                for _ in range(1, k):
                    next_index = next_avail[current]
                    if next_index >= start + n_points:
                        break
                    count += 1
                    current = next_index
                if count < k:
                    continue
                wrap_gap = T[start] + L - T[current]
                if wrap_gap >= d:
                    return True
            return False
        
        low, high = 0, 2 * side
        while low < high:
            mid = (low + high + 1) // 2
            if feasible(mid):
                low = mid
            else:
                high = mid - 1
        return low