from typing import List

# This solution uses `sortedcontainers.SortedList` for O(log n) updates.
# If this library is not available, the fallback list-based implementation 
# will be used, which might be too slow for worst-case inputs.
try:
    from sortedcontainers import SortedList
except ImportError:
    import bisect
    class SortedList:
        def __init__(self, iterable=None):
            self._list = sorted(list(iterable)) if iterable is not None else []
        def add(self, value): bisect.insort_left(self._list, value)
        def remove(self, value):
            i = bisect.bisect_left(self._list, value)
            if i < len(self._list) and self._list[i] == value: self._list.pop(i)
            else: raise ValueError(f"{value} not in list")
        def index(self, value):
            i = bisect.bisect_left(self._list, value)
            if i < len(self._list) and self._list[i] == value: return i
            raise ValueError(f"{value} not in list")
        def bisect_right(self, value): return bisect.bisect_right(self._list, value)
        def __getitem__(self, i): return self._list[i]
        def __len__(self): return len(self._list)
        def __bool__(self): return bool(self._list)

class FenwickTree:
    """A 0-indexed Fenwick Tree."""
    def __init__(self, size):
        self.tree = [0] * (size + 1)

    def update(self, i, delta):
        i += 1
        while i < len(self.tree):
            self.tree[i] += delta
            i += i & (-i)

    def query(self, i):
        i += 1
        s = 0
        while i > 0:
            s += self.tree[i]
            i -= i & (-i)
        return s

    def query_range(self, i, j):
        if i > j:
            return 0
        res = self.query(j)
        if i > 0:
            res -= self.query(i - 1)
        return res

class Solution:
    def numberOfAlternatingGroups(self, colors: List[int], queries: List[List[int]]) -> List[int]:
        self.n = len(colors)
        
        self.bit1 = FenwickTree(self.n + 1)
        self.bit2 = FenwickTree(self.n + 1)
        
        self.is_alt = [(colors[i] != colors[(i + 1) % self.n]) for i in range(self.n)]
        
        self.zeros = SortedList([i for i, val in enumerate(self.is_alt) if val == 0])

        if not self.zeros:
            if self.n > 0:
                self._update_run_len(self.n, 1)
        else:
            num_zeros = len(self.zeros)
            for i in range(num_zeros):
                z1 = self.zeros[i - 1]
                z2 = self.zeros[i]
                length = (z2 - z1 + self.n) % self.n - 1
                if length > 0:
                    self._update_run_len(length, 1)
        
        ans = []
        for query in queries:
            q_type = query[0]
            if q_type == 1:
                k = query[1]
                sum_t2 = self.bit2.query_range(k - 1, self.n)
                sum_t1 = self.bit1.query_range(k - 1, self.n)
                count = sum_t2 - k * sum_t1
                ans.append(count)
            else:
                idx, c = query[1], query[2]
                if colors[idx] == c:
                    continue
                
                for i in [(idx - 1 + self.n) % self.n, idx]:
                    old_alt = self.is_alt[i]
                    if i == (idx - 1 + self.n) % self.n:
                        new_alt = (colors[i] != c)
                    else:
                        new_alt = (c != colors[(i + 1) % self.n])
                    
                    if old_alt != new_alt:
                        self._update_is_alt_at(i, new_alt)
                
                colors[idx] = c
                
        return ans

    def _update_run_len(self, length, delta):
        self.bit1.update(length, delta)
        self.bit2.update(length, delta * (length + 2))

    def _update_is_alt_at(self, i, new_val):
        self.is_alt[i] = new_val

        if new_val == 0:  # 1 -> 0: a run of 1s is split
            if not self.zeros:
                self._update_run_len(self.n, -1)
                if self.n - 1 > 0: self._update_run_len(self.n - 1, 1)
            else:
                idx = self.zeros.bisect_right(i)
                num_zeros = len(self.zeros)
                z_next = self.zeros[idx % num_zeros]
                z_prev = self.zeros[idx - 1]
                
                old_len = (z_next - z_prev + self.n) % self.n - 1
                if old_len > 0: self._update_run_len(old_len, -1)
                
                new_len1 = (i - z_prev + self.n) % self.n - 1
                if new_len1 > 0: self._update_run_len(new_len1, 1)
                
                new_len2 = (z_next - i + self.n) % self.n - 1
                if new_len2 > 0: self._update_run_len(new_len2, 1)
            
            self.zeros.add(i)
        
        else:  # 0 -> 1: two runs of 1s merge
            idx = self.zeros.index(i)
            num_zeros = len(self.zeros)

            if num_zeros == 1:
                if self.n - 1 > 0: self._update_run_len(self.n - 1, -1)
                if self.n > 0: self._update_run_len(self.n, 1)
            else:
                z_prev = self.zeros[idx - 1]
                z_next = self.zeros[(idx + 1) % num_zeros]
                
                len1 = (i - z_prev + self.n) % self.n - 1
                if len1 > 0: self._update_run_len(len1, -1)
                
                len2 = (z_next - i + self.n) % self.n - 1
                if len2 > 0: self._update_run_len(len2, -1)
                
                new_len = (z_next - z_prev + self.n) % self.n - 1
                if new_len > 0: self._update_run_len(new_len, 1)
            
            self.zeros.remove(i)