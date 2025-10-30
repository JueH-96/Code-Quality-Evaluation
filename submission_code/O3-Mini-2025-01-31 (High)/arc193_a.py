import sys, bisect
 
INF = 10**18
 
# --- Iterative Segment Tree for range minimum query ---
class SegTree:
    def __init__(self, data):
        n = len(data)
        self.n = 1
        while self.n < n:
            self.n *= 2
        self.size = self.n
        self.tree = [INF]*(2*self.n)
        for i in range(n):
            self.tree[self.n + i] = data[i]
        for i in range(self.n - 1, 0, -1):
            self.tree[i] = min(self.tree[2*i], self.tree[2*i+1])
 
    def query(self, l, r):
        # returns min in interval [l, r)
        if r <= l:
            return INF
        res = INF
        l += self.n
        r += self.n
        while l < r:
            if l & 1:
                res = min(res, self.tree[l])
                l += 1
            if r & 1:
                r -= 1
                res = min(res, self.tree[r])
            l //= 2
            r //= 2
        return res
 
def main():
    data = sys.stdin.buffer.read().split()
    if not data:
        return
    it = iter(data)
    n = int(next(it))
    W_arr = [0]*n
    for i in range(n):
        W_arr[i] = int(next(it))
    L_arr = [0]*n
    R_arr = [0]*n
    max_coord = 0
    for i in range(n):
        L_arr[i] = int(next(it))
        R_arr[i] = int(next(it))
        max_coord = max(max_coord, L_arr[i], R_arr[i])
    M = max_coord + 2   # coordinates 0..max_coord+1
    # --- Precompute global min_right: for each x, min_right[x] = min{ W[i] : L[i] >= x } ---
    min_right = [INF]*M
    for i in range(n):
        coord = L_arr[i]
        if W_arr[i] < min_right[coord]:
            min_right[coord] = W_arr[i]
    for x in range(M-2, -1, -1):
        if min_right[x+1] < min_right[x]:
            min_right[x] = min_right[x+1]
 
    # --- Precompute global min_left: for each x, min_left[x] = min{ W[i] : R[i] <= x } ---
    min_left = [INF]*M
    for i in range(n):
        coord = R_arr[i]
        if W_arr[i] < min_left[coord]:
            min_left[coord] = W_arr[i]
    for x in range(1, M):
        if min_left[x-1] < min_left[x]:
            min_left[x] = min_left[x-1]
 
    # --- Build a segment tree on the R–coordinate: for each r we want the minimum W among vertices with R == r.
    segRdata = [INF]*M
    for i in range(n):
        r = R_arr[i]
        if W_arr[i] < segRdata[r]:
            segRdata[r] = W_arr[i]
    segRtree = SegTree(segRdata)
    # segRtree.query(a,b) returns min W among vertices with R in [a,b)
 
    # --- Build sorted list of vertices by L; each element is (L, R, W) ---
    sortedVerts = []
    for i in range(n):
        sortedVerts.append((L_arr[i], R_arr[i], W_arr[i]))
    sortedVerts.sort(key=lambda x: x[0])
 
    # --- Helper functions ---
    def candidate2(sIdx, tIdx):
        # candidate2: one intermediate vertex x that is disjoint with both s and t.
        Ls, Rs, Ws = L_arr[sIdx], R_arr[sIdx], W_arr[sIdx]
        Lt, Rt, Wt = L_arr[tIdx], R_arr[tIdx], W_arr[tIdx]
        # If the intervals are disjoint then a direct edge exists.
        if Rs < Lt or Rt < Ls:
            return Ws + Wt
        threshold_right = max(Rs, Rt)
        cand_right = min_right[threshold_right + 1] if (threshold_right+1) < M else INF
        threshold_left = min(Ls, Lt)
        cand_left = min_left[threshold_left - 1] if (threshold_left-1) >= 0 else INF
        bestCand = min(cand_right, cand_left)
        return Ws + Wt + bestCand
 
    def candidate3(sIdx, tIdx):
        # candidate3: using two intermediates. We assume (swapping if needed) that L[s] <= L[t].
        Ls, Rs, Ws = L_arr[sIdx], R_arr[sIdx], W_arr[sIdx]
        Lt, Rt, Wt = L_arr[tIdx], R_arr[tIdx], W_arr[tIdx]
        if Ls > Lt:
            sIdx, tIdx = tIdx, sIdx
            Ls, Rs, Ws = L_arr[sIdx], R_arr[sIdx], W_arr[sIdx]
            Lt, Rt, Wt = L_arr[tIdx], R_arr[tIdx], W_arr[tIdx]
        # We search for candidate x (the first extra) among vertices with:
        #   L(x) > Rs   and   L(x) <= Rt.
        lo_val = Rs + 1
        hi_val = Rt
        low_index = bisect.bisect_left(sortedVerts, (lo_val, -INF, -INF))
        high_index = bisect.bisect_right(sortedVerts, (hi_val, INF, INF))
        best_pair = INF
        # For each candidate x we “simulate” choosing a y as the second intermediate.
        # If x’s L is less than Lt then we require y with R in [Ls, L(x))
        # Otherwise (if L(x) >= Lt) we pick among y with R in [Ls, Lt).
        for i in range(low_index, high_index):
            Lx, Rx, Wx = sortedVerts[i]
            if Lx < Lt:
                extra = segRtree.query(Ls, Lx)
            else:
                extra = segRtree.query(Ls, Lt)
            if extra == INF:
                continue
            candidate_val = Wx + extra
            if candidate_val < best_pair:
                best_pair = candidate_val
        if best_pair == INF:
            return INF
        return Ws + Wt + best_pair
 
    q = int(next(it))
    outLines = []
    for _ in range(q):
        s_q = int(next(it))
        t_q = int(next(it))
        sIdx = s_q - 1
        tIdx = t_q - 1
        Ls, Rs, Ws = L_arr[sIdx], R_arr[sIdx], W_arr[sIdx]
        Lt, Rt, Wt = L_arr[tIdx], R_arr[tIdx], W_arr[tIdx]
        # If disjoint then answer = W[s] + W[t]
        if Rs < Lt or Rt < Ls:
            outLines.append(str(Ws + Wt))
        else:
            cand2 = candidate2(sIdx, tIdx)
            cand3 = candidate3(sIdx, tIdx)
            best = min(cand2, cand3)
            outLines.append(str(best) if best < INF else "-1")
    sys.stdout.write("
".join(outLines))
 
if __name__ == '__main__':
    main()