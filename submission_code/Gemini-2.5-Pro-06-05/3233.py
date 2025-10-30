class Solution:
    def maxPartitionsAfterOperations(self, s: str, k: int) -> int:
        
        class SegTree:
            def __init__(self, arr):
                self.n = len(arr)
                self.tree = [0] * (4 * self.n)
                if self.n > 0:
                    self._build(arr, 0, 0, self.n - 1)

            def _build(self, arr, v, tl, tr):
                if tl == tr:
                    self.tree[v] = arr[tl]
                else:
                    tm = (tl + tr) // 2
                    self._build(arr, 2 * v + 1, tl, tm)
                    self._build(arr, 2 * v + 2, tm + 1, tr)
                    self.tree[v] = max(self.tree[2 * v + 1], self.tree[2 * v + 2])

            def query(self, l, r):
                if l > r or self.n == 0:
                    return 0
                return self._query(0, 0, self.n - 1, l, r)

            def _query(self, v, tl, tr, l, r):
                if l > r:
                    return 0
                if l == tl and r == tr:
                    return self.tree[v]
                tm = (tl + tr) // 2
                left_max = self._query(2 * v + 1, tl, tm, l, min(r, tm))
                right_max = self._query(2 * v + 2, tm + 1, tr, max(l, tm + 1), r)
                return max(left_max, right_max)
            
            def update(self, pos, new_val):
                if self.n == 0: return
                self._update(0, 0, self.n - 1, pos, new_val)

            def _update(self, v, tl, tr, pos, new_val):
                if tl == tr:
                    self.tree[v] = new_val
                else:
                    tm = (tl + tr) // 2
                    if pos <= tm:
                        self._update(2 * v + 1, tl, tm, pos, new_val)
                    else:
                        self._update(2 * v + 2, tm + 1, tr, pos, new_val)
                    self.tree[v] = max(self.tree[2 * v + 1], self.tree[2 * v + 2])

        n = len(s)
        if k == 26:
            return 1

        def compute_next_cut(target_distinct):
            cuts = [n] * n
            counts = {}
            right = 0
            for left in range(n):
                while right < n:
                    char_r = s[right]
                    is_new = char_r not in counts
                    if is_new and len(counts) == target_distinct:
                        cuts[left] = right
                        break
                    counts[char_r] = counts.get(char_r, 0) + 1
                    right += 1
                else:
                    cuts[left] = n
                
                char_l = s[left]
                counts[char_l] -= 1
                if counts[char_l] == 0:
                    del counts[char_l]
            return cuts

        # partition s[i:j], next starts at j
        part_ends_k = [n] * n
        for i in range(n):
            distinct = set()
            for j in range(i, n):
                distinct.add(s[j])
                if len(distinct) > k:
                    part_ends_k[i] = j
                    break
        
        g = [0] * (n + 1)
        for i in range(n - 1, -1, -1):
            g[i] = 1 + g[part_ends_k[i]]

        if k == 26: return g[0]
            
        next_cut_k1 = compute_next_cut(k + 1)
        next_cut_k2 = compute_next_cut(k + 2)
        
        next_occ = [[n] * 26 for _ in range(n + 1)]
        for i in range(n - 1, -1, -1):
            for char_code in range(26):
                next_occ[i][char_code] = next_occ[i+1][char_code]
            next_occ[i][ord(s[i]) - ord('a')] = i
            
        rmq_g = SegTree(g)
        
        dp0 = [0] * (n + 1)
        rmq_dp0 = SegTree([0] * (n + 1))
        
        for i in range(n - 1, -1, -1):
            # Term A: partition s[i:j] without change, solve s[j:] with 1 change
            j_part_end = part_ends_k[i]
            max_future_dp = rmq_dp0.query(i + 1, j_part_end)
            term_A = 1 + max_future_dp
            
            # Term B: partition s[i:j] with change, solve s[j:] with 0 changes (g)
            term_B = 0
            
            j_start_fixable = next_cut_k1[i]
            if j_start_fixable < n:
                distinct_chars = set(s[i : j_start_fixable + 1])
                j_end_fixable = next_cut_k2[i]

                for char in distinct_chars:
                    char_code = ord(char) - ord('a')
                    p1 = next_occ[i][char_code]
                    p2 = next_occ[p1 + 1][char_code]
                    
                    query_l = j_start_fixable
                    query_r = min(j_end_fixable, p2) - 1
                    
                    if query_l <= query_r:
                        max_g = rmq_g.query(query_l, query_r)
                        term_B = max(term_B, 1 + max_g)

            dp0[i] = max(term_A, term_B)
            rmq_dp0.update(i, dp0[i])
            
        return dp0[0]