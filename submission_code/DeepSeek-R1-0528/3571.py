class SegmentTree:
    def __init__(self, n):
        self.n = n
        self.size = 1
        while self.size < n:
            self.size *= 2
        self.data = [0] * (2 * self.size)
    
    def update(self, i, val):
        i += self.size
        if val > self.data[i]:
            self.data[i] = val
            while i > 1:
                i //= 2
                self.data[i] = max(self.data[2*i], self.data[2*i+1])
    
    def query(self, l, r):
        if l > r:
            return 0
        l += self.size
        r += self.size
        res = 0
        while l <= r:
            if l % 2 == 1:
                res = max(res, self.data[l])
                l += 1
            if r % 2 == 0:
                res = max(res, self.data[r])
                r -= 1
            l //= 2
            r //= 2
        return res

class Solution:
    def maxPathLength(self, coordinates: List[List[int]], k: int) -> int:
        n = len(coordinates)
        points = []
        for i, (x, y) in enumerate(coordinates):
            points.append((x, y, i))
        
        all_ys = [y for x, y, i in points]
        sorted_ys = sorted(set(all_ys))
        comp = {}
        for idx, y_val in enumerate(sorted_ys):
            comp[y_val] = idx + 1
        m = len(sorted_ys)
        
        points_asc = sorted(points, key=lambda p: (p[0], p[1]))
        seg_back = SegmentTree(m)
        dp_backward = [0] * n
        
        i = 0
        while i < n:
            j = i
            current_x = points_asc[i][0]
            group = []
            while j < n and points_asc[j][0] == current_x:
                group.append(points_asc[j])
                j += 1
            for x, y, idx_val in group:
                j_comp = comp[y]
                max_val = seg_back.query(0, j_comp - 2)
                dp_backward[idx_val] = max_val + 1
            for x, y, idx_val in group:
                j_comp = comp[y]
                seg_back.update(j_comp - 1, dp_backward[idx_val])
            i = j
        
        points_desc = sorted(points, key=lambda p: (-p[0], -p[1]))
        seg_forward = SegmentTree(m)
        dp_forward = [0] * n
        
        i = 0
        while i < n:
            j = i
            current_x = points_desc[i][0]
            group = []
            while j < n and points_desc[j][0] == current_x:
                group.append(points_desc[j])
                j += 1
            for x, y, idx_val in group:
                j_comp = comp[y]
                max_val = seg_forward.query(j_comp, m - 1)
                dp_forward[idx_val] = max_val + 1
            for x, y, idx_val in group:
                j_comp = comp[y]
                seg_forward.update(j_comp - 1, dp_forward[idx_val])
            i = j
        
        result = dp_backward[k] + dp_forward[k] - 1
        return result