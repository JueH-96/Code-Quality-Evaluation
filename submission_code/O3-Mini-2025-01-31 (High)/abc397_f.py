# YOUR CODE HERE
def main():
    import sys,sys
    data = sys.stdin.read().strip().split()
    if not data: 
        return
    it = iter(data)
    try:
        n = int(next(it))
    except StopIteration:
        return
    A = [int(next(it)) for _ in range(n)]
    
    # Using 1-indexed positions.
    # Build a dictionary "occ" mapping each number to the list of positions (1-indexed) at which it occurs.
    occ = {}
    for i, a in enumerate(A, start=1):
        if a not in occ:
            occ[a] = []
        occ[a].append(i)
        
    # Total distinct number count.
    T = len(occ)
    singles = 0
    for a in occ:
        if len(occ[a]) == 1:
            singles += 1
    # Base score: for a single-occurrence number the score is always 1,
    # for a number with at least 2 occurrences the guaranteed score is 2.
    # Thus base_sum = (# singles) + 2*(# non-singles) = 2*T - (# singles)
    base_sum = 2 * T - singles
    
    # Now for bonus: Only numbers with frequency ≥ 3 can achieve the extra bonus point.
    # For a number x with freq ≥ 3, let f = first occurrence, s = second occurrence, L = last occurrence.
    # A bonus is earned if we choose i (the first cut, i.e. end of segment1) and j (the second cut)
    # so that i is in [f+1, s] and j is in [s+1, L].
    #
    # Our allowed splits (in 1-indexed terms) are: i in [1, n-2] and j in [2, n-1] with i < j.
    # We'll “clamp” our intervals accordingly.
    #
    # For each x with freq ≥ 3, we create a bonus rectangle:
    #    i ∈ [L_i, R_i] where L_i = f+1 and R_i = s   (and then clamp to [1, n-2])
    #    j ∈ [L_j, R_j] where L_j = s+1 and R_j = L   (and then clamp to [2, n-1])
    events = []  # Each event: (j_event, L_i, R_i, delta)
    M = n - 2  # number of valid i split positions (i=1..n-2)
    for a, pos_list in occ.items():
        if len(pos_list) >= 3:
            f = pos_list[0]
            s = pos_list[1]
            L_ = pos_list[-1]
            Li = f + 1
            Ri = s
            Lj = s + 1
            Rj = L_
            # Clamp to allowable ranges:
            if Li < 1:
                Li = 1
            if Ri > n - 2:
                Ri = n - 2
            if Lj < 2:
                Lj = 2
            if Rj > n - 1:
                Rj = n - 1
            if Li <= Ri and Lj <= Rj:
                events.append((Lj, Li, Ri, 1))
                events.append((Rj + 1, Li, Ri, -1))
    
    events.sort(key=lambda x: x[0])
    
    # We'll implement a lazy segment tree that supports range-add and range-max query.
    # The tree covers indices 0 .. M-1, corresponding to split positions i = 1 .. M
    class LazySegTree:
        __slots__ = "n", "size", "data", "lazy"
        def __init__(self, n):
            self.n = n
            size = 1
            while size < n:
                size *= 2
            self.size = size
            self.data = [0] * (2 * size)
            self.lazy = [0] * (2 * size)
        def _apply(self, i, val):
            self.data[i] += val
            if i < self.size:
                self.lazy[i] += val
        # Range update: add val to all indices in [l, r] (0-indexed, inclusive)
        def update(self, l, r, val):
            l0 = l + self.size
            r0 = r + self.size
            l2, r2 = l0, r0
            while l0 <= r0:
                if (l0 & 1) == 1:
                    self._apply(l0, val)
                    l0 += 1
                if (r0 & 1) == 0:
                    self._apply(r0, val)
                    r0 -= 1
                l0 //= 2
                r0 //= 2
            i = l2
            while i > 1:
                i //= 2
                self.data[i] = max(self.data[2*i], self.data[2*i+1]) + self.lazy[i]
            i = r2
            while i > 1:
                i //= 2
                self.data[i] = max(self.data[2*i], self.data[2*i+1]) + self.lazy[i]
        def _push(self, i):
            h = self.size.bit_length() - 1
            for d in range(h, 0, -1):
                j = i >> d
                if self.lazy[j] != 0:
                    self._apply(2*j, self.lazy[j])
                    self._apply(2*j+1, self.lazy[j])
                    self.lazy[j] = 0
        # Range query: maximum in [l, r] inclusive (0-indexed)
        def query(self, l, r):
            l += self.size
            r += self.size
            self._push(l)
            self._push(r)
            res = -10**9
            while l <= r:
                if (l & 1) == 1:
                    if self.data[l] > res:
                        res = self.data[l]
                    l += 1
                if (r & 1) == 0:
                    if self.data[r] > res:
                        res = self.data[r]
                    r -= 1
                l //= 2
                r //= 2
            return res

    segtree = LazySegTree(M)  # indices 0..M-1 correspond to valid i=1..M
    bonus_opt = 0
    ev_ptr = 0
    # We'll “sweep” over possible second cut j.
    # Valid j are 1-indexed from 2 to n-1.
    for j in range(2, n):
        # Process all events with event j coordinate equal to current j.
        while ev_ptr < len(events) and events[ev_ptr][0] == j:
            j_ev, Li, Ri, delta = events[ev_ptr]
            # Our segtree uses 0-indexed positions for i, so positions i=Li..Ri in 1-indexed become indices Li-1..Ri-1.
            segtree.update(Li - 1, Ri - 1, delta)
            ev_ptr += 1
        # For the current j, we are allowed to choose i in [1, min(j-1, M)]
        max_i = min(j - 1, M)
        if max_i > 0:
            # In segtree (0-indexed), these positions are indices 0 .. max_i-1.
            cand = segtree.query(0, max_i - 1)
            if cand > bonus_opt:
                bonus_opt = cand
    if bonus_opt < 0:
        bonus_opt = 0
    ans = base_sum + bonus_opt
    sys.stdout.write(str(ans))
    
if __name__ == '__main__':
    main()