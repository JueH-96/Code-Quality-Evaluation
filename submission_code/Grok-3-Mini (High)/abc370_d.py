import sys
sys.setrecursionlimit(1000000)

class SegmentTree:
    def __init__(self, size, init_val=1):
        self.size = size
        self.has_one = [False] * (4 * size)
        self.min_one_idx = [-1] * (4 * size)
        self.max_one_idx = [-1] * (4 * size)
        self.build(1, 1, size, init_val)

    def build(self, node, start, end, val):
        if start == end:
            if val == 1:
                self.has_one[node] = True
                self.min_one_idx[node] = start
                self.max_one_idx[node] = start
            else:
                self.has_one[node] = False
                self.min_one_idx[node] = -1
                self.max_one_idx[node] = -1
        else:
            mid = (start + end) // 2
            self.build(node * 2, start, mid, val)
            self.build(node * 2 + 1, mid + 1, end, val)
            left = node * 2
            right = node * 2 + 1
            self.has_one[node] = self.has_one[left] or self.has_one[right]
            if self.has_one[left] and self.has_one[right]:
                self.min_one_idx[node] = min(self.min_one_idx[left], self.min_one_idx[right])
                self.max_one_idx[node] = max(self.max_one_idx[left], self.max_one_idx[right])
            elif self.has_one[left]:
                self.min_one_idx[node] = self.min_one_idx[left]
                self.max_one_idx[node] = self.max_one_idx[left]
            elif self.has_one[right]:
                self.min_one_idx[node] = self.min_one_idx[right]
                self.max_one_idx[node] = self.max_one_idx[right]
            else:
                self.min_one_idx[node] = -1
                self.max_one_idx[node] = -1

    def update(self, node, start, end, idx, new_val):
        if start == end:
            if new_val == 1:
                self.has_one[node] = True
                self.min_one_idx[node] = idx
                self.max_one_idx[node] = idx
            else:
                self.has_one[node] = False
                self.min_one_idx[node] = -1
                self.max_one_idx[node] = -1
        else:
            mid = (start + end) // 2
            if idx <= mid:
                self.update(node * 2, start, mid, idx, new_val)
            else:
                self.update(node * 2 + 1, mid + 1, end, idx, new_val)
            left = node * 2
            right = node * 2 + 1
            self.has_one[node] = self.has_one[left] or self.has_one[right]
            if self.has_one[left] and self.has_one[right]:
                self.min_one_idx[node] = min(self.min_one_idx[left], self.min_one_idx[right])
                self.max_one_idx[node] = max(self.max_one_idx[left], self.max_one_idx[right])
            elif self.has_one[left]:
                self.min_one_idx[node] = self.min_one_idx[left]
                self.max_one_idx[node] = self.max_one_idx[left]
            elif self.has_one[right]:
                self.min_one_idx[node] = self.min_one_idx[right]
                self.max_one_idx[node] = self.max_one_idx[right]
            else:
                self.min_one_idx[node] = -1
                self.max_one_idx[node] = -1

    def get(self, idx):
        return self._get(1, 1, self.size, idx)

    def _get(self, node, start, end, idx):
        if start == end:
            return 1 if self.has_one[node] else 0
        else:
            mid = (start + end) // 2
            if idx <= mid:
                return self._get(node * 2, start, mid, idx)
            else:
                return self._get(node * 2 + 1, mid + 1, end, idx)

    def find_max_lt(self, pos):
        return self._find_max_lt(1, 1, self.size, pos)

    def _find_max_lt(self, node, start, end, pos):
        if start >= pos or not self.has_one[node]:
            return -1
        if end < pos:
            return self.max_one_idx[node]
        mid = (start + end) // 2
        left_res = self._find_max_lt(node * 2, start, mid, pos)
        if left_res != -1:
            return left_res
        if mid + 1 < pos and self.has_one[node * 2 + 1]:
            right_res = self._find_max_lt(node * 2 + 1, mid + 1, end, pos)
            if right_res != -1:
                return right_res
        return -1

    def find_min_gt(self, pos):
        return self._find_min_gt(1, 1, self.size, pos)

    def _find_min_gt(self, node, start, end, pos):
        if end <= pos or not self.has_one[node]:
            return -1
        if start > pos:
            return self.min_one_idx[node]
        mid = (start + end) // 2
        right_res = self._find_min_gt(node * 2 + 1, mid + 1, end, pos)
        if right_res != -1:
            return right_res
        if mid > pos and self.has_one[node * 2]:
            left_res = self._find_min_gt(node * 2, start, mid, pos)
            return left_res
        return -1

# Read input
data = sys.stdin.read().split()
index = 0
H = int(data[index])
index += 1
W = int(data[index])
index += 1
Q = int(data[index])
index += 1

# Create segment trees
row_trees = [SegmentTree(W) for _ in range(H)]
column_trees = [SegmentTree(H) for _ in range(W)]

# Initial number of walls
remaining_walls = H * W

# Process each query
for _ in range(Q):
    R = int(data[index])
    index += 1
    C = int(data[index])
    index += 1
    
    # Check if wall at (R, C)
    if row_trees[R-1].get(C) == 1:
        # Destroy the wall at (R, C)
        row_trees[R-1].update(1, 1, W, C, 0)  # Update in row tree
        column_trees[C-1].update(1, 1, H, R, 0)  # Update in column tree
        remaining_walls -= 1
    else:
        # No wall at (R, C), destroy nearest in four directions
        # Left
        j_left = row_trees[R-1].find_max_lt(C)
        if j_left != -1:
            row_trees[R-1].update(1, 1, W, j_left, 0)
            column_trees[j_left-1].update(1, 1, H, R, 0)
            remaining_walls -= 1
        # Right
        j_right = row_trees[R-1].find_min_gt(C)
        if j_right != -1:
            row_trees[R-1].update(1, 1, W, j_right, 0)
            column_trees[j_right-1].update(1, 1, H, R, 0)
            remaining_walls -= 1
        # Up
        i_up = column_trees[C-1].find_max_lt(R)
        if i_up != -1:
            row_trees[i_up-1].update(1, 1, W, C, 0)
            column_trees[C-1].update(1, 1, H, i_up, 0)
            remaining_walls -= 1
        # Down
        i_down = column_trees[C-1].find_min_gt(R)
        if i_down != -1:
            row_trees[i_down-1].update(1, 1, W, C, 0)
            column_trees[C-1].update(1, 1, H, i_down, 0)
            remaining_walls -= 1

# Output the number of remaining walls
print(remaining_walls)