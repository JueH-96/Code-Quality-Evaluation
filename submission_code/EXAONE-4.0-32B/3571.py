class Solution:
    def maxPathLength(self, coordinates, k: int) -> int:
        n = len(coordinates)
        points = []
        for i, (x, y) in enumerate(coordinates):
            points.append((x, y, i))
        
        all_ys = sorted(set(y for x, y in coordinates))
        comp_map = {y_val: idx+1 for idx, y_val in enumerate(all_ys)}
        maxY = len(all_ys)
        
        points_sorted_left = sorted(points, key=lambda p: (p[0], p[1]))
        dp_end = [0] * n
        tree_left = [0] * (maxY + 1)
        
        i = 0
        batch = []
        prev_x = None
        while i < n:
            x, y, idx = points_sorted_left[i]
            if x != prev_x:
                for (yy, jdx, val) in batch:
                    cy = comp_map[yy]
                    self.update_fw(tree_left, cy, val, maxY)
                batch = []
                prev_x = x
            cy = comp_map[y]
            if cy - 1 >= 1:
                q = self.query_fw(tree_left, cy - 1)
            else:
                q = 0
            dp_end[idx] = q + 1
            batch.append((y, idx, dp_end[idx]))
            i += 1
        
        for (yy, jdx, val) in batch:
            cy = comp_map[yy]
            self.update_fw(tree_left, cy, val, maxY)
        
        points_sorted_right = sorted(points, key=lambda p: (-p[0], p[1]))
        dp_start = [0] * n
        tree_right = [0] * (maxY + 1)
        
        i = 0
        batch = []
        prev_x = None
        while i < n:
            x, y, idx = points_sorted_right[i]
            if x != prev_x:
                for (yy, jdx, val) in batch:
                    rev_y = maxY - comp_map[yy] + 1
                    self.update_fw(tree_right, rev_y, val, maxY)
                batch = []
                prev_x = x
            rev_y = maxY - comp_map[y] + 1
            if rev_y - 1 >= 1:
                q = self.query_fw(tree_right, rev_y - 1)
            else:
                q = 0
            dp_start[idx] = q + 1
            batch.append((y, idx, dp_start[idx]))
            i += 1
        
        for (yy, jdx, val) in batch:
            rev_y = maxY - comp_map[yy] + 1
            self.update_fw(tree_right, rev_y, val, maxY)
        
        target = k
        return dp_end[target] + dp_start[target] - 1

    def update_fw(self, tree, index, value, size):
        while index <= size:
            if value > tree[index]:
                tree[index] = value
            index += index & -index

    def query_fw(self, tree, index):
        res = 0
        while index > 0:
            if tree[index] > res:
                res = tree[index]
            index -= index & -index
        return res