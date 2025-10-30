from typing import List

class Solution:
    def numberOfAlternatingGroups(self, colors: List[int], queries: List[List[int]]) -> List[int]:
        n = len(colors)
        # Build the adjacency array A: A[i]=1 if colors[i] != colors[i-1], else 0
        A = [0] * n
        for i in range(n):
            A[i] = 1 if colors[i] != colors[(i - 1) % n] else 0

        # Fenwick / BIT for zeros in A
        class BIT:
            def __init__(self, size):
                self.n = size
                self.tree = [0] * (size + 1)
                # highest power of two <= n, for find_by_order
                self.max_log = (size).bit_length() - 1

            # add value v at index i (1-based)
            def add(self, i, v):
                while i <= self.n:
                    self.tree[i] += v
                    i += i & -i

            # sum of [1..i] (1-based)
            def sum(self, i):
                s = 0
                while i > 0:
                    s += self.tree[i]
                    i -= i & -i
                return s

            # find smallest idx i such that sum(1..i) >= k, assume 1 <= k <= sum(n)
            def find_by_order(self, k):
                pos = 0
                # binary lift
                for d in range(self.max_log, -1, -1):
                    nxt = pos + (1 << d)
                    if nxt <= self.n and self.tree[nxt] < k:
                        k -= self.tree[nxt]
                        pos = nxt
                return pos + 1

        # BIT0 tracks zeros in A: bit0[i]=1 if A[i-1]==0
        bit0 = BIT(n)
        for i in range(n):
            if A[i] == 0:
                bit0.add(i + 1, 1)

        # Two BITs to track run-length frequencies and run-length sums
        # run lengths range 1..n
        freqBIT = BIT(n)
        sumBIT = BIT(n)

        # helper: distance between a,b exclusive in circular array of length n
        def dist(a, b):
            # number of positions strictly between a and b in the circle
            if b > a:
                return b - a - 1
            else:
                return b + n - a - 1

        # initialize runs from initial zeros
        zeros = [i for i in range(n) if A[i] == 0]
        k0 = len(zeros)
        if k0 == 0:
            # no zeros => one run of length n
            freqBIT.add(n, 1)
            sumBIT.add(n, n)
        else:
            # runs between consecutive zeros
            for j in range(k0 - 1):
                L = dist(zeros[j], zeros[j + 1])
                if L > 0:
                    freqBIT.add(L, 1)
                    sumBIT.add(L, L)
            # wrap-around run
            L = dist(zeros[-1], zeros[0])
            if L > 0:
                freqBIT.add(L, 1)
                sumBIT.add(L, L)

        res = []

        # Handlers for flipping A[pos] from 0->1 or 1->0
        def handle_zero_to_one(pos):
            # pos: 0-based index
            p_idx = pos + 1
            total_z = bit0.sum(n)
            s0 = bit0.sum(p_idx)
            if total_z == 1:
                # one zero at pos; old run length = n-1
                oldL = n - 1
                freqBIT.add(oldL, -1)
                sumBIT.add(oldL, -oldL)
                # remove zero
                bit0.add(p_idx, -1)
                # new run length = n
                newL = n
                freqBIT.add(newL, 1)
                sumBIT.add(newL, newL)
            else:
                # find neighbors of pos in zero-list
                # next zero after pos
                if s0 < total_z:
                    nxt_idx = bit0.find_by_order(s0 + 1)
                else:
                    nxt_idx = bit0.find_by_order(1)
                # prev zero before pos
                if s0 > 1:
                    prv_idx = bit0.find_by_order(s0 - 1)
                else:
                    prv_idx = bit0.find_by_order(total_z)
                r = nxt_idx - 1
                l = prv_idx - 1
                # lengths of the two runs separated by pos
                L1 = dist(l, pos)
                L2 = dist(pos, r)
                if L1 > 0:
                    freqBIT.add(L1, -1)
                    sumBIT.add(L1, -L1)
                if L2 > 0:
                    freqBIT.add(L2, -1)
                    sumBIT.add(L2, -L2)
                # remove zero at pos
                bit0.add(p_idx, -1)
                # merged run length
                newL = L1 + L2 + 1
                freqBIT.add(newL, 1)
                sumBIT.add(newL, newL)

        def handle_one_to_zero(pos):
            p_idx = pos + 1
            total_z = bit0.sum(n)
            if total_z == 0:
                # no zeros; old run length = n
                oldL = n
                freqBIT.add(oldL, -1)
                sumBIT.add(oldL, -oldL)
                # insert zero
                bit0.add(p_idx, 1)
                # new single run length = n-1
                newL = n - 1
                freqBIT.add(newL, 1)
                sumBIT.add(newL, newL)
            else:
                # find neighbors where pos will split run
                s0 = bit0.sum(p_idx)
                # next zero after pos
                if s0 < total_z:
                    nxt_idx = bit0.find_by_order(s0 + 1)
                else:
                    nxt_idx = bit0.find_by_order(1)
                # prev zero before pos
                if s0 > 0:
                    prv_idx = bit0.find_by_order(s0)
                else:
                    prv_idx = bit0.find_by_order(total_z)
                r = nxt_idx - 1
                l = prv_idx - 1
                # old run length
                oldL = dist(l, r)
                freqBIT.add(oldL, -1)
                sumBIT.add(oldL, -oldL)
                # insert zero
                bit0.add(p_idx, 1)
                # new left and right runs
                L1 = dist(l, pos)
                L2 = dist(pos, r)
                if L1 > 0:
                    freqBIT.add(L1, 1)
                    sumBIT.add(L1, L1)
                if L2 > 0:
                    freqBIT.add(L2, 1)
                    sumBIT.add(L2, L2)

        # process queries
        for q in queries:
            if q[0] == 1:
                # count alternating groups of size k
                k = q[1]
                t = k - 1
                # number of runs of length >= t
                total_runs = freqBIT.sum(n) - freqBIT.sum(t - 1)
                # sum of those run lengths
                total_len = sumBIT.sum(n) - sumBIT.sum(t - 1)
                # each run of length L contributes (L - (t - 1)) windows
                ans = total_len - (t - 1) * total_runs
                res.append(ans)
            else:
                # update color
                idx, cnew = q[1], q[2]
                if colors[idx] == cnew:
                    continue
                colors[idx] = cnew
                # two adjacency bits affected: A[idx] and A[idx+1 mod n]
                for d in (idx, (idx + 1) % n):
                    old = A[d]
                    prev_col = colors[(d - 1) % n]
                    new = 1 if colors[d] != prev_col else 0
                    if old == new:
                        continue
                    # flip A[d]
                    A[d] = new
                    if new == 1:
                        # 0 -> 1
                        handle_zero_to_one(d)
                    else:
                        # 1 -> 0
                        handle_one_to_zero(d)

        return res