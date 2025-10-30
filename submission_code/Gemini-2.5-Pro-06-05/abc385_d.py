import sys
from collections import defaultdict
from bisect import bisect_left, bisect_right

# It's a good practice for deep recursion, although the segment tree depth
# is logarithmic, so it might not be strictly necessary.
sys.setrecursionlimit(2 * 10**5 + 100)

class SegmentTree:
    """A segment tree for sum queries and point updates.
    It is specifically tailored to find and clear elements."""
    def __init__(self, size):
        self.size = size
        if size == 0:
            self.tree = []
        else:
            self.tree = [0] * (4 * size)
            self._build(0, 0, size - 1)

    def _build(self, node, start, end):
        if start == end:
            self.tree[node] = 1
            return
        mid = (start + end) // 2
        self._build(2 * node + 1, start, mid)
        self._build(2 * node + 2, mid + 1, end)
        self.tree[node] = self.tree[2 * node + 1] + self.tree[2 * node + 2]

    def _update(self, node, start, end, idx, val):
        if start == end:
            self.tree[node] = val
            return
        mid = (start + end) // 2
        if start <= idx <= mid:
            self._update(2 * node + 1, start, mid, idx, val)
        else:
            self._update(2 * node + 2, mid + 1, end, idx, val)
        self.tree[node] = self.tree[2 * node + 1] + self.tree[2 * node + 2]
    
    def update(self, idx, val):
        if self.size == 0: return
        self._update(0, 0, self.size - 1, idx, val)

    def _find_and_clear_recursive(self, node, start, end, l, r, found_indices):
        if self.tree[node] == 0 or start > r or end < l:
            return
        if start == end:
            found_indices.append(start)
            self.tree[node] = 0
            return
        
        mid = (start + end) // 2
        self._find_and_clear_recursive(2 * node + 1, start, mid, l, r, found_indices)
        self._find_and_clear_recursive(2 * node + 2, mid + 1, end, l, r, found_indices)
        self.tree[node] = self.tree[2 * node + 1] + self.tree[2 * node + 2]

    def find_and_clear(self, l, r):
        if self.size == 0 or l > r: return []
        found_indices = []
        self._find_and_clear_recursive(0, 0, self.size - 1, l, r, found_indices)
        return found_indices

def solve():
    """
    Main function to solve the problem.
    """
    input = sys.stdin.readline
    N, M, S_x, S_y = map(int, input().split())
    
    houses = [tuple(map(int, input().split())) for _ in range(N)]
    moves = []
    for _ in range(M):
        d, c = input().split()
        moves.append((d, int(c)))

    # --- Pre-computation ---
    houses_by_x = defaultdict(list)
    houses_by_y = defaultdict(list)
    for i in range(N):
        x, y = houses[i]
        houses_by_x[x].append((y, i))
        houses_by_y[y].append((x, i))

    house_info = [{} for _ in range(N)]
    
    for x in houses_by_x:
        houses_by_x[x].sort()
        for j, (y, house_id) in enumerate(houses_by_x[x]):
            info = house_info[house_id]
            info['x'] = x
            info['y'] = y
            info['idx_x'] = j

    for y in houses_by_y:
        houses_by_y[y].sort()
        for j, (x, house_id) in enumerate(houses_by_y[y]):
            house_info[house_id]['idx_y'] = j
            
    segtree_by_x = {x: SegmentTree(len(houses_by_x[x])) for x in houses_by_x}
    segtree_by_y = {y: SegmentTree(len(houses_by_y[y])) for y in houses_by_y}

    # --- Simulation ---
    current_x, current_y = S_x, S_y
    found_count = 0
    
    for direction, dist in moves:
        prev_x, prev_y = current_x, current_y
        
        if direction == 'U':
            current_y += dist
            x = prev_x
            if x in segtree_by_x:
                y_list = houses_by_x[x]
                st_x = segtree_by_x[x]
                y1, y2 = prev_y, current_y
                start_idx = bisect_left(y_list, (y1, -1))
                end_idx = bisect_right(y_list, (y2, N + 1))
                
                found_indices = st_x.find_and_clear(start_idx, end_idx - 1)
                found_count += len(found_indices)
                for idx in found_indices:
                    hy, hid = y_list[idx]
                    info = house_info[hid]
                    st_y = segtree_by_y[info['y']]
                    st_y.update(info['idx_y'], 0)
        
        elif direction == 'D':
            current_y -= dist
            x = prev_x
            if x in segtree_by_x:
                y_list = houses_by_x[x]
                st_x = segtree_by_x[x]
                y1, y2 = current_y, prev_y
                start_idx = bisect_left(y_list, (y1, -1))
                end_idx = bisect_right(y_list, (y2, N + 1))

                found_indices = st_x.find_and_clear(start_idx, end_idx - 1)
                found_count += len(found_indices)
                for idx in found_indices:
                    hy, hid = y_list[idx]
                    info = house_info[hid]
                    st_y = segtree_by_y[info['y']]
                    st_y.update(info['idx_y'], 0)

        elif direction == 'R':
            current_x += dist
            y = prev_y
            if y in segtree_by_y:
                x_list = houses_by_y[y]
                st_y = segtree_by_y[y]
                x1, x2 = prev_x, current_x
                start_idx = bisect_left(x_list, (x1, -1))
                end_idx = bisect_right(x_list, (x2, N + 1))

                found_indices = st_y.find_and_clear(start_idx, end_idx - 1)
                found_count += len(found_indices)
                for idx in found_indices:
                    hx, hid = x_list[idx]
                    info = house_info[hid]
                    st_x = segtree_by_x[info['x']]
                    st_x.update(info['idx_x'], 0)

        elif direction == 'L':
            current_x -= dist
            y = prev_y
            if y in segtree_by_y:
                x_list = houses_by_y[y]
                st_y = segtree_by_y[y]
                x1, x2 = current_x, prev_x
                start_idx = bisect_left(x_list, (x1, -1))
                end_idx = bisect_right(x_list, (x2, N + 1))

                found_indices = st_y.find_and_clear(start_idx, end_idx - 1)
                found_count += len(found_indices)
                for idx in found_indices:
                    hx, hid = x_list[idx]
                    info = house_info[hid]
                    st_x = segtree_by_x[info['x']]
                    st_x.update(info['idx_x'], 0)

    print(current_x, current_y, found_count)

if __name__ == "__main__":
    solve()