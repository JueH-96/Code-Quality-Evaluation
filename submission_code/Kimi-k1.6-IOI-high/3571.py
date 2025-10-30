class SegmentTree:
    def __init__(self, size):
        self.n = size
        self.size = 1
        while self.size < self.n:
            self.size <<= 1
        self.data = [-1] * (2 * self.size)
    
    def update(self, pos, value):
        pos += self.size
        if self.data[pos] >= value:
            return
        self.data[pos] = value
        while pos > 1:
            pos >>= 1
            new_val = max(self.data[2*pos], self.data[2*pos+1])
            if self.data[pos] == new_val:
                break
            self.data[pos] = new_val
    
    def query(self, l, r):
        res = -1
        l += self.size
        r += self.size
        while l <= r:
            if l % 2 == 1:
                res = max(res, self.data[l])
                l += 1
            if r % 2 == 0:
                res = max(res, self.data[r])
                r -= 1
            l >>= 1
            r >>= 1
        return res

class Solution:
    def maxPathLength(self, coordinates: List[List[int]], k: int) -> int:
        coords = [tuple(coord) for coord in coordinates]
        k_coord = coords[k]
        
        sorted_forward = sorted(coords, key=lambda x: (x[0], x[1]))
        sorted_backward = sorted(coords, key=lambda x: (-x[0], x[1]))
        
        all_y = [y for x, y in coords]
        sorted_unique_y = sorted(list(set(all_y)))
        if not sorted_unique_y:
            return 1
        compression = {y: idx for idx, y in enumerate(sorted_unique_y)}
        m = len(sorted_unique_y)
        max_compressed_y_index = m - 1
        
        dp_forward = {}
        st_forward = SegmentTree(m)
        
        groups_forward = []
        prev_x = None
        current_group = []
        for x, y in sorted_forward:
            if x != prev_x:
                if current_group:
                    groups_forward.append(current_group)
                current_group = [(x, y)]
                prev_x = x
            else:
                current_group.append((x, y))
        if current_group:
            groups_forward.append(current_group)
        
        for group in groups_forward:
            for x, y in group:
                y_compressed = compression[y]
                if y_compressed > 0:
                    max_val = st_forward.query(0, y_compressed - 1)
                else:
                    max_val = -1
                dp_forward[(x, y)] = max_val + 1 if max_val != -1 else 1
            for x, y in group:
                y_compressed = compression[y]
                current_dp = dp_forward[(x, y)]
                st_forward.update(y_compressed, current_dp)
        
        dp_backward = {}
        st_backward = SegmentTree(m)
        
        groups_backward = []
        prev_x = None
        current_group = []
        for x, y in sorted_backward:
            if x != prev_x:
                if current_group:
                    groups_backward.append(current_group)
                current_group = [(x, y)]
                prev_x = x
            else:
                current_group.append((x, y))
        if current_group:
            groups_backward.append(current_group)
        
        for group in groups_backward:
            for x, y in group:
                y_compressed = compression[y]
                left = y_compressed + 1
                right = max_compressed_y_index
                if left > right:
                    max_val = -1
                else:
                    max_val = st_backward.query(left, right)
                dp_backward[(x, y)] = max_val + 1 if max_val != -1 else 1
            for x, y in group:
                y_compressed = compression[y]
                current_dp = dp_backward[(x, y)]
                st_backward.update(y_compressed, current_dp)
        
        df = dp_forward.get(k_coord, 1)
        db = dp_backward.get(k_coord, 1)
        return max(df, db, df + db - 1)