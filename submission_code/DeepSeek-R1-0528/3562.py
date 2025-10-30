import sys
sys.setrecursionlimit(300000)

class SegTree:
    def __init__(self, size):
        self.n = size
        self.tree = [None] * (4 * self.n)
    
    def update_point(self, pos, state):
        self._update(1, 1, self.n, pos, state)
    
    def _update(self, idx, l, r, pos, state):
        if l == r:
            if self.tree[idx] is None or self._better(state, self.tree[idx]):
                self.tree[idx] = state
            return
        mid = (l + r) // 2
        if pos <= mid:
            self._update(idx*2, l, mid, pos, state)
        else:
            self._update(idx*2+1, mid+1, r, pos, state)
        left_val = self.tree[idx*2]
        right_val = self.tree[idx*2+1]
        self.tree[idx] = self._best(left_val, right_val)
    
    def query_range(self, ql, qr):
        if ql > qr:
            return None
        return self._query(1, 1, self.n, ql, qr)
    
    def _query(self, idx, l, r, ql, qr):
        if ql <= l and r <= qr:
            return self.tree[idx]
        mid = (l + r) // 2
        left_res = None
        right_res = None
        if ql <= mid:
            left_res = self._query(idx*2, l, mid, ql, min(qr, mid))
        if qr > mid:
            right_res = self._query(idx*2+1, mid+1, r, max(ql, mid+1), qr)
        return self._best(left_res, right_res)
    
    def _better(self, a, b):
        if b is None:
            return a is not None
        if a is None:
            return False
        if a[0] > b[0]:
            return True
        if a[0] == b[0] and a[1] < b[1]:
            return True
        return False

    def _best(self, a, b):
        if a is None:
            return b
        if b is None:
            return a
        if a[0] > b[0] or (a[0] == b[0] and a[1] < b[1]):
            return a
        return b

class Solution:
    def maximumWeight(self, intervals: List[List[int]]) -> List[int]:
        n = len(intervals)
        all_coords = []
        intervals_with_idx = []
        for idx, (l, r, w) in enumerate(intervals):
            intervals_with_idx.append((l, r, w, idx))
            all_coords.append(l)
            all_coords.append(r)
        
        coords = sorted(set(all_coords))
        mapping = {x: i+1 for i, x in enumerate(coords)}
        size_coords = len(coords)
        
        intervals_with_idx.sort(key=lambda x: x[1])
        
        seg_trees = [None]
        for _ in range(4):
            seg_trees.append(SegTree(size_coords))
        
        global_best = [None] * 5
        
        for l, r, w, orig_idx in intervals_with_idx:
            disc_l = mapping[l]
            disc_r = mapping[r]
            for k in range(1, 5):
                if k == 1:
                    state = (w, (orig_idx,))
                    if global_best[1] is None or self.better(state, global_best[1]):
                        global_best[1] = state
                    seg_trees[1].update_point(disc_r, state)
                else:
                    prev_state = seg_trees[k-1].query_range(1, disc_l - 1)
                    if prev_state is None:
                        continue
                    new_weight = prev_state[0] + w
                    new_indices = tuple(sorted(prev_state[1] + (orig_idx,)))
                    state = (new_weight, new_indices)
                    if global_best[k] is None or self.better(state, global_best[k]):
                        global_best[k] = state
                    seg_trees[k].update_point(disc_r, state)
        
        max_weight = 0
        candidates = []
        for k in range(1, 5):
            if global_best[k] is not None:
                if global_best[k][0] > max_weight:
                    max_weight = global_best[k][0]
                    candidates = [global_best[k][1]]
                elif global_best[k][0] == max_weight:
                    candidates.append(global_best[k][1])
        
        best_candidate = min(candidates) if candidates else []
        return list(best_candidate)
    
    def better(self, a, b):
        if b is None:
            return a is not None
        if a is None:
            return False
        if a[0] > b[0]:
            return True
        if a[0] == b[0] and a[1] < b[1]:
            return True
        return False