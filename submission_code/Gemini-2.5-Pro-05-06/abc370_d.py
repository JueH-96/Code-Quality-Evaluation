import sys

# Fast I/O
input = sys.stdin.readline

class SegmentTree:
    def __init__(self, size):
        # size is the number of leaves, e.g., W or H. Constraints: size >= 1.
        self.size = size
        # For N leaves, array size up to 4N is safe for 1-indexed segment tree nodes
        self.tree = [0] * (4 * self.size) 
        self._build(1, 0, self.size - 1)

    def _build(self, node_idx, L, R):
        if L == R:
            self.tree[node_idx] = 1 # Initially all walls exist
            return
        mid = (L + R) // 2
        self._build(2 * node_idx, L, mid)
        self._build(2 * node_idx + 1, mid + 1, R)
        self.tree[node_idx] = self.tree[2 * node_idx] + self.tree[2 * node_idx + 1]

    def get_val(self, target_idx):
        # Assumes 0 <= target_idx < self.size
        return self._get_val_recursive(1, 0, self.size - 1, target_idx)

    def _get_val_recursive(self, node_idx, L, R, target_idx):
        if L == R:
            return self.tree[node_idx]
        mid = (L + R) // 2
        if target_idx <= mid:
            return self._get_val_recursive(2 * node_idx, L, mid, target_idx)
        else:
            return self._get_val_recursive(2 * node_idx + 1, mid + 1, R, target_idx)

    def update(self, target_idx, val):
        # Assumes 0 <= target_idx < self.size
        self._update_recursive(1, 0, self.size - 1, target_idx, val)

    def _update_recursive(self, node_idx, L, R, target_idx, val):
        if L == R:
            self.tree[node_idx] = val
            return
        mid = (L + R) // 2
        if target_idx <= mid:
            self._update_recursive(2 * node_idx, L, mid, target_idx, val)
        else:
            self._update_recursive(2 * node_idx + 1, mid + 1, R, target_idx, val)
        self.tree[node_idx] = self.tree[2 * node_idx] + self.tree[2 * node_idx + 1]

    def find_smallest_idx_in_range(self, query_L, query_R):
        if query_L > query_R: # Original query range is empty/invalid
            return None
        
        # Clamp query range to be within ST bounds [0, self.size-1]
        actual_query_L = max(0, query_L)
        actual_query_R = min(self.size - 1, query_R)

        if actual_query_L > actual_query_R: # Clamped range is empty/invalid
            return None
            
        return self._find_smallest_recursive(1, 0, self.size - 1, actual_query_L, actual_query_R)

    def _find_smallest_recursive(self, node_idx, L, R, query_L, query_R):
        if self.tree[node_idx] == 0: 
            return None
        if L > query_R or R < query_L: 
            return None
        if L == R: 
            return L 
        
        mid = (L + R) // 2
        res_left = self._find_smallest_recursive(2 * node_idx, L, mid, query_L, query_R)
        if res_left is not None:
            return res_left
        
        res_right = self._find_smallest_recursive(2 * node_idx + 1, mid + 1, R, query_L, query_R)
        return res_right

    def find_largest_idx_in_range(self, query_L, query_R):
        if query_L > query_R:
            return None
        
        actual_query_L = max(0, query_L)
        actual_query_R = min(self.size - 1, query_R)

        if actual_query_L > actual_query_R:
            return None

        return self._find_largest_recursive(1, 0, self.size - 1, actual_query_L, actual_query_R)

    def _find_largest_recursive(self, node_idx, L, R, query_L, query_R):
        if self.tree[node_idx] == 0:
            return None
        if L > query_R or R < query_L:
            return None
        if L == R:
            return L
            
        mid = (L + R) // 2
        res_right = self._find_largest_recursive(2 * node_idx + 1, mid + 1, R, query_L, query_R)
        if res_right is not None:
            return res_right
            
        res_left = self._find_largest_recursive(2 * node_idx, L, mid, query_L, query_R)
        return res_left

def solve():
    H, W, Q = map(int, input().split())

    row_trees = [SegmentTree(W) for _ in range(H)]
    col_trees = [SegmentTree(H) for _ in range(W)]
    
    num_walls = H * W

    for _ in range(Q):
        Rq, Cq = map(int, input().split())
        r, c = Rq - 1, Cq - 1 # 0-indexed

        walls_to_destroy_coords = set()

        if row_trees[r].get_val(c) == 1:
            walls_to_destroy_coords.add((r, c))
        else:
            # Up: In col_trees[c], search rows 0 to r-1. Find largest index.
            idx = col_trees[c].find_largest_idx_in_range(0, r - 1)
            if idx is not None:
                walls_to_destroy_coords.add((idx, c))
            
            # Down: In col_trees[c], search rows r+1 to H-1. Find smallest index.
            idx = col_trees[c].find_smallest_idx_in_range(r + 1, H - 1)
            if idx is not None:
                walls_to_destroy_coords.add((idx, c))

            # Left: In row_trees[r], search cols 0 to c-1. Find largest index.
            idx = row_trees[r].find_largest_idx_in_range(0, c - 1)
            if idx is not None:
                walls_to_destroy_coords.add((r, idx))

            # Right: In row_trees[r], search cols c+1 to W-1. Find smallest index.
            idx = row_trees[r].find_smallest_idx_in_range(c + 1, W - 1)
            if idx is not None:
                walls_to_destroy_coords.add((r, idx))
        
        for wr, wc in walls_to_destroy_coords:
            if row_trees[wr].get_val(wc) == 1: # Check if wall still exists
                row_trees[wr].update(wc, 0)
                col_trees[wc].update(wr, 0)
                num_walls -= 1
                
    print(num_walls)

solve()