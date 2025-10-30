from collections import defaultdict

class FenwickTreeMax:
    def __init__(self, size):
        self.size = size
        self.tree = [0] * (self.size + 2)  # 1-based indexing

    def update(self, idx, value):
        while idx <= self.size:
            if value > self.tree[idx]:
                self.tree[idx] = value
            else:
                break  # No need to update further if the current value is not larger
            idx += idx & -idx

    def query(self, idx):
        res = 0
        while idx > 0:
            res = max(res, self.tree[idx])
            idx -= idx & -idx
        return res

class Solution:
    def maxPathLength(self, coordinates: List[List[int]], k: int) -> int:
        points = [tuple(p) for p in coordinates]
        target_point = tuple(coordinates[k])

        # Preprocess y ranks for forward and backward passes
        all_y = [y for x, y in points]
        unique_y = sorted(list(set(all_y)))
        y_rank_forward = {y: i + 1 for i, y in enumerate(unique_y)}  # 1-based for Fenwick
        forward_size = len(unique_y)

        unique_y_reversed = sorted(list(set(all_y)), reverse=True)
        y_rank_backward = {y: i + 1 for i, y in enumerate(unique_y_reversed)}
        backward_size = len(unique_y_reversed)

        # Compute dp_forward
        sorted_forward = sorted(points, key=lambda p: (p[0], p[1]))
        dp_forward = defaultdict(int)
        fenwick_forward = FenwickTreeMax(forward_size)
        for p in sorted_forward:
            x, y = p
            rank = y_rank_forward[y]
            max_val = fenwick_forward.query(rank - 1)
            dp_forward[p] = max_val + 1
            fenwick_forward.update(rank, dp_forward[p])

        # Compute dp_backward
        sorted_backward = sorted(points, key=lambda p: (-p[0], -p[1]))
        grouped_backward = []
        current_x = None
        current_group = []
        for p in sorted_backward:
            if p[0] != current_x:
                if current_group:
                    grouped_backward.append(current_group)
                current_group = [p]
                current_x = p[0]
            else:
                current_group.append(p)
        if current_group:
            grouped_backward.append(current_group)
        dp_backward = defaultdict(int)
        fenwick_backward = FenwickTreeMax(backward_size)
        for group in grouped_backward:
            for p in group:
                x, y = p
                rank = y_rank_backward[y]
                max_val = fenwick_backward.query(rank - 1)
                dp_backward[p] = max_val + 1
            for p in group:
                x, y = p
                rank = y_rank_backward[y]
                fenwick_backward.update(rank, dp_backward[p])

        # Compute the maximum path including target_point
        if target_point not in dp_forward or target_point not in dp_backward:
            return 1  # This should not happen as the point is present
        res = dp_forward[target_point] + dp_backward[target_point] - 1
        return res