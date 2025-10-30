class SegmentTreeMax:
    def __init__(self, size):
        self.n = 1
        while self.n < size:
            self.n <<= 1
        self.size = self.n
        self.tree = [0] * (2 * self.n)

    def update(self, pos, value):
        pos += self.n - 1  # convert to 0-based index
        if self.tree[pos] >= value:
            return
        self.tree[pos] = value
        i = pos >> 1  # parent
        while i >= 1:
            new_val = max(self.tree[2 * i], self.tree[2 * i + 1])
            if self.tree[i] == new_val:
                break
            self.tree[i] = new_val
            i >>= 1

    def query_range(self, l, r):
        if l > r:
            return 0
        l += self.n - 1
        r += self.n - 1
        res = 0
        while l <= r:
            if l % 2 == 1:
                res = max(res, self.tree[l])
                l += 1
            if r % 2 == 0:
                res = max(res, self.tree[r])
                r -= 1
            l >>= 1
            r >>= 1
        return res

    def query(self, up_to):
        return self.query_range(1, up_to)

class Solution:
    def maxPathLength(self, coordinates: List[List[int]], k: int) -> int:
        # Compress y-values
        ys = set(y for x, y in coordinates)
        sorted_ys = sorted(ys)
        y_rank = {y: i+1 for i, y in enumerate(sorted_ys)}
        max_rank = len(sorted_ys)
        if max_rank == 0:
            return 1

        # Create original indices mapping
        original_indices = {(x, y): i for i, (x, y) in enumerate(coordinates)}

        # Compute L values
        sorted_for_L = sorted(coordinates, key=lambda p: (p[0], p[1]))
        groups = []
        current_x = None
        current_group = []
        for p in sorted_for_L:
            if p[0] != current_x:
                if current_group:
                    groups.append(current_group)
                current_x = p[0]
                current_group = [p]
            else:
                current_group.append(p)
        if current_group:
            groups.append(current_group)

        st_L = SegmentTreeMax(max_rank)
        L = [0] * len(coordinates)
        for group in groups:
            updates = []
            for p in group:
                x_p, y_p = p
                rank = y_rank[y_p]
                max_prev = st_L.query(rank - 1)
                current_L = max_prev + 1
                original_index = original_indices[(x_p, y_p)]
                L[original_index] = current_L
                updates.append((rank, current_L))
            for rank, val in updates:
                st_L.update(rank, val)

        # Compute R values
        sorted_for_R = sorted(coordinates, key=lambda p: (-p[0], p[1]))
        groups_R = []
        current_x_R = None
        current_group_R = []
        for p in sorted_for_R:
            if p[0] != current_x_R:
                if current_group_R:
                    groups_R.append(current_group_R)
                current_x_R = p[0]
                current_group_R = [p]
            else:
                current_group_R.append(p)
        if current_group_R:
            groups_R.append(current_group_R)

        st_R = SegmentTreeMax(max_rank)
        R = [0] * len(coordinates)
        for group in groups_R:
            updates_R = []
            for p in group:
                x_p, y_p = p
                rank = y_rank[y_p]
                max_next = st_R.query_range(rank + 1, max_rank)
                current_R = max_next + 1
                original_index = original_indices[(x_p, y_p)]
                R[original_index] = current_R
                updates_R.append((rank, current_R))
            for rank, val in updates_R:
                st_R.update(rank, val)

        # Get the k-th point's L and R
        k_point = coordinates[k]
        k_original_index = original_indices[(k_point[0], k_point[1])]
        return L[k_original_index] + R[k_original_index] - 1