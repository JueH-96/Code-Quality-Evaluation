def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    M = int(input_data[1])
    K = int(input_data[2])
    A = list(map(int, input_data[3:]))

    # ----------------------------------------------------------------
    # Explanation of the approach:
    #
    # We have N candidates, each currently with A[i] votes. There are K total votes
    # and ∑A[i] have already been cast/observed, so R = K - sum(A) votes remain
    # to be distributed in unknown ways among the N candidates.
    #
    # A candidate i is guaranteed to be among the "winners" if, no matter how the
    # remaining R votes are allocated among the other (N-1) candidates, the number
    # of candidates with strictly more votes than i is < M.
    #
    # We want, for each i, the smallest nonnegative X ≤ R such that giving exactly
    # X of the remaining R votes to candidate i (and distributing the other R-X
    # among the others in the worst possible way for i) still ensures i is in the top M.
    # If no such X exists, we print -1; if i is *already* guaranteed with X=0,
    # we print 0.
    #
    # Key idea: If we fix i's final total T = A[i]+X, the worst case is that each other
    # candidate j tries to surpass T by receiving as many of the leftover votes (R-X)
    # as needed. Candidate j can surpass i iff A[j] + (R - X) > T. We only need to
    # let fewer than M of them do that. Equivalently the number of j who *can* surpass i
    # must be < M.
    #
    # A direct check for each i would be O(N^2), too big. Instead, we rely on sorting
    # and careful counting:
    #
    # Let B = sorted list of A. For a particular i with value A[i] = x:
    #  1) Count how many j have A[j] > x + R  (these are "guaranteed to surpass i"
    #     because even if i got all R leftover votes, i's total is x+R, and j's total
    #     is > x+R already). Let that count be G. If G >= M, i can never be in top M -> -1.
    #  2) Exclude those G from consideration; the others are "potential surpassers."
    #     Among those potential ones, define the set S of candidates j (excluding i itself).
    #  3) With X=0, if G plus the size of S is still < M, then i is guaranteed even at X=0 -> 0.
    #  4) Otherwise, we find the minimal X that reduces the number of "able to surpass i"
    #     below M-G. One can show a candidate j can surpass i if
    #           A[j] + (R - X) > x + X   =>   2X < A[j] + R - x
    #     The borderline depends monotonically on A[j].
    #
    # A convenient way to implement the "kth largest in a range" logic (to find
    # the threshold) is via an Order Statistic Tree or Fenwick (BIT) with coordinate
    # compression. We do the following steps:
    #
    #   - Coordinate-compress all A[i] so that we can store frequencies in a Fenwick tree.
    #   - For each i in original order:
    #       * Let x = A[i]. G = number of j with A[j] > x+R (by Fenwick query).
    #         If G >= M: answer -1.
    #         Else remove i's own frequency from Fenwick (so we don't count i among its own surpassers).
    #         Let size = number of j (in Fenwick) with A[j] in (x-R, x+R] range, i.e. "potential".
    #         If G + size < M, answer 0 (already safe).
    #         Else find T = (M - 1 - G). We need to ensure at most T in that set can still surpass i.
    #         That amounts to finding the (T+1)-th largest A[j] in that set, call it pivot.
    #         Then the critical X is  floor( (pivot - x - 1 + R) // 2 ) + 1 .
    #         Clamp it to [0..R]. If it <= R, that's the answer, else -1.
    #         Re-insert i into Fenwick.
    #
    # This yields an O(N log N) solution.
    #
    # Below is an implementation of this plan.
    #
    # ----------------------------------------------------------------

    # Edge cases quick check:
    #  - If M > N, problem statement says M ≤ N, so not an issue.
    #  - If sum(A) = K, R=0, no more votes left. Then everyone is locked in. We can
    #    simply check how many are strictly above each A[i] and see if < M -> 0, else -1.

    # 1) Compute R
    sA = sum(A)
    R = K - sA

    # Pre-check: if R == 0, answers are simpler:
    # For each i, count how many j have A[j] > A[i]. If that count < M, answer=0, else -1.
    # We can do that quickly with a sorted array. Then we are done.
    if R == 0:
        # Sort the distinct values of A, then for each i do a binary search
        # to count how many are strictly greater than A[i].
        # If that count < M => 0 else -1
        B = sorted(A)
        ans = []
        import bisect
        Nminus1 = N - 1
        for x in A:
            # number strictly greater = N - index of first x+1 or more
            idx = bisect.bisect_right(B, x)  # first position where B[idx] > x
            # count of elements > x is N - idx
            if N - idx < M:
                ans.append(0)
            else:
                ans.append(-1)
        print(" ".join(map(str, ans)))
        return

    # Otherwise, we proceed with the Fenwick-based solution.
    # 2) Build arrays for coordinate compression:
    #    We want to be able to query:
    #        G = number of j with A[j] > x+R
    #        size = number of j with x-R < A[j] <= x+R   [but we'll refine endpoints slightly]
    #    We'll keep everything in a Fenwick tree with compressed indices.

    # We'll store all possible "pivot" values we might query: A[i], A[i]+R+1, A[i]-R (or A[i]-R+1).
    # Actually we only need to be able to do rank queries for x+R+1 and x-R+1, plus the A[i] themselves.

    vals = []
    for v in A:
        vals.append(v)           # needed to re-insert / remove
        vals.append(v + R + 1)   # for G = # > x+R
        vals.append(v - R + 1)   # for lower bound checks

    # Remove duplicates and sort
    vals = list(set(vals))
    vals.sort()

    # A small helper for coordinate compression: get index of x in vals
    # (we use bisect_left because vals is sorted and unique).
    import bisect
    def compress(x):
        # index in the sorted list "vals"
        return bisect.bisect_left(vals, x)

    # 3) Fenwick tree for frequencies of the original A array.
    #    We will store freq of each compressed A[i].
    #    Then we can do prefix sums to count # <= something, and # > something.
    class Fenwick:
        def __init__(self, n):
            self.n = n
            self.fw = [0]*(n+1)
        def update(self, i, delta):
            # i in [0..n-1], internally use i+1
            i += 1
            while i <= self.n:
                self.fw[i] += delta
                i += i & -i
        def query(self, i):
            # sum from 0..i
            s = 0
            while i>0:
                s += self.fw[i]
                i -= i & -i
            return s
        def range_query(self, l, r):
            if r<l: return 0
            return self.query(r) - self.query(l-1)
        def find_kth_in_range_desc(self, l, r, k):
            """
            Find the k-th largest element within [l..r] by frequency.
            k=1 means the largest in [l..r].
            
            We'll do a standard Fenwick binary search but restricted to the range [l..r].
            We'll find how many elements are in [r..r], [r-1..r], etc. 
            However, implementing a partial-range "k-th largest" directly is tricky in 
            a single Fenwick. We can do a standard "find_kth" for all 0..r, then 
            subtract those 0..(l-1), but that doesn't easily yield the index. 
            
            Instead we do a manual approach:
              total_in_range = range_query(l,r). We want the k-th from the 'end' of that range.
              That is the (total_in_range - k + 1)-th from the start of that range in ascending order.

            So let rank_asc = total_in_range - k + 1. Then we do "find_kth_in_range_asc(l,r, rank_asc)".
            We'll implement "find_kth_in_prefix" to find the smallest idx such that prefix_sum(idx) >= some_value.
            Then we do a small logic to skip below 'l'. We'll do a left-exclusion by zeroing out the freq 
            of [0..l-1] temporarily or do an offset. For performance, we won't physically zero anything out; 
            we can do a custom function that queries partial sums but also excludes partial sums from [0..l-1]. 
            
            To keep the code simpler, we can do a standard "find_kth_in_prefix" on the entire Fenwick, 
            let idxA = find_kth_in_prefix( prefix_needed ), 
            but we must ensure that prefix_needed accounts only for the frequencies in [l..r].
            
            We'll do a small loop searching from high to low, skipping out-of-range indices. 
            However, that can degrade to O(n). 
            
            Because M can be up to N and N up to 200k, we do need an O(log N) approach. 
            
            We'll implement a 2-Fenwicks technique:
              - fenAll: contains freq for all indices
              - fenLeft: contains freq for [0..l-1] so we can do "range" = fenAll.query(x) - fenLeft.query(x). 
            Then the partial sum for [l..x] is fenAll.query(x) - fenLeft.query(x). 
            We'll do a binary search for the smallest x such that that partial sum >= rank_asc. 
            
            Implementation detail: we only need it once per candidate, so 2 Fenwicks might still be too big to rebuild. 
            
            Another simpler approach: we only do this once to find the pivot. Because M <= N= 200k, 
            doing such an O(log N) search 200k times is feasible. That is 2e5 * log(2e5) ~ 2. 
            
            We'll implement two Fenwicks: fenAll (the main one), fenLeft (the partial for prefix [0..l-1]), so that
            partial(i) = fenAll.query(i) - fenLeft.query(i). Then searching for partial(i) >= rank_asc. 
            
            Let's do that.
            """
            total = self.range_query(l, r)
            # If k>total, not found => shouldn't happen if caller is correct
            rank_asc = total - (k - 1)  # "k-th largest" => rank_asc in ascending
            # We'll do a standard fenwicksum-based search for fenAll, but we subtract fenLeft on the fly.

            # The typical Fenwick binary searching approach (for prefix sums):
            #   we keep an index = 0, then from the largest power of two down to 1,
            #   we try stepping up if fenwAll[index+step] < needed, etc. but we must also subtract fenLeft.
            # We'll define a function partial_sum(i) = fenAll.query(i) - fenLeft.query(i).

            # We'll do a separate Fenwicks structure approach. Because we didn't actually build fenLeft,
            # let's do it now on the fly. That means building fenLeft for each query i => that would be O(N) again. 
            # Instead, we will do a direct "prefix subtraction" approach:
            #
            #   partial_sum(mid) = fenwAll.query(mid) - fenwAllLeft,
            # but that "fenwAllLeft" is not just a single number, we need the sum of freq in [0..l-1].
            # Actually sum in [0..l-1] is constant. Then sum in [0..mid] minus sum in [0..l-1] => we want >= rank_asc. 
            #
            # We'll do a binary search for the smallest mid s.t. range_query(l, mid) >= rank_asc.
            # We can do range_query(l, mid) = fenw.query(mid) - fenw.query(l-1). We'll do a standard
            # "Fenwicksum-based search" but we can't simply do the typical approach because we always have to subtract fenw.query(l-1). 
            #
            # We'll do a standard integer binary search from l..r. That is O(log(r-l)), which is O(log N). That is still fine. 
            #
            # Implementation below: we do a plain while low<=high binary search:
            fenw = self

            want = rank_asc
            low, high = l, r
            ans_idx = r  # we'll store the smallest index meeting the criterion
            base_left_sum = fenw.query(l-1) if l>0 else 0

            while low <= high:
                mid = (low+high)//2
                s_inrange = fenw.query(mid) - base_left_sum
                if s_inrange >= want:
                    ans_idx = mid
                    high = mid-1
                else:
                    low = mid+1
            return ans_idx


    # 4) Build Fenwicks
    # We'll store all frequencies of A in fenAll
    # Because for the partial-range search we do a standard binary search, one Fenwicks is enough.

    # Make the Fenwicks
    compA = [compress(x) for x in A]  # compressed index for each candidate's A[i]
    fenAll = Fenwick(len(vals))

    # fill fenAll
    for cidx in compA:
        fenAll.update(cidx, 1)

    # We'll need a function to remove one occurrence of A[i] from fenAll, then re-add it.
    def remove_candidate(i):
        fenAll.update(compA[i], -1)
    def add_candidate(i):
        fenAll.update(compA[i], 1)

    ans = [0]*N

    import math

    # Precompute all pos[i] = index of A[i] in sorted B was not strictly needed, 
    # but we do want them for quick remove/add:
    # We already have that in compA.

    # Helper to get count > x
    #   => # of elements with index > compress(x). Actually we want strictly > x,
    #   so we do pos = compress(x+1)
    #   count > x = total - fenAll.query(pos-1)
    def count_greater_than(x):
        # we want # of elements strictly greater than x
        # compress(x+1) => all elements >= x+1
        cp = compress(x+1)
        return fenAll.query(len(vals)) - fenAll.query(cp-1 if cp>0 else 0)

    # Helper to get count in (l..r], i.e. strictly > l and <= r:
    #   # <= r minus # <= l
    def count_in_range(l, r):
        if r<l:
            return 0
        if l<0: l=0
        cp_r = compress(r)  # first index where vals[idx]>=r => we want <= r, so we'll adjust soon
        # but be careful: compress(r) = index where vals[idx]>= r
        # We want all vals <= r. If vals[cp_r] == r, we want to include it => i.e. we want to query up to cp_r if vals[cp_r] == r
        # so if r == vals[cp_r], we want to go up to cp_r indeed. If r < vals[cp_r], we want cp_r-1. 
        if cp_r< len(vals) and vals[cp_r] == r:
            hi = cp_r
        else:
            hi = cp_r - 1
        if hi<0:
            return 0

        cp_l = compress(l)  # index where vals[idx]>=l
        # if l == vals[cp_l], we want > l => that means we skip cp_l if l is exactly in the array. 
        # Actually we want (l..r] => that is >l and <= r. So we exclude l exactly. 
        # If l is in vals, we should not count it. 
        # So effectively we want # <= r minus # <= l. 
        # # <= l is query up to index compress(l) if l in vals. Let's do it systematically:
        # # <= x is fenAll.query( idx ) where idx is ??? 
        # We'll do hi as above. For l, we do compress(l). If vals[cp_l] == l, we want to subtract fenAll.query(cp_l). 
        # else we want fenAll.query(cp_l-1).
        if cp_l< len(vals) and vals[cp_l] == l:
            lo = cp_l
        else:
            lo = cp_l - 1
        if lo<0:
            cnt_le_l = 0
        else:
            cnt_le_l = fenAll.query(lo)

        cnt_le_r = fenAll.query(hi)
        return max(0, cnt_le_r - cnt_le_l)

    # We also need the “count > x+R” done as G, so G = count_greater_than(x+R).
    # Then the potential set S is those with A[j] in (x-R, x+R], but we'll exclude i by removing it temporarily.

    # Let us implement and go:
    for i in range(N):
        x = A[i]
        # 1) Count G = # j with A[j] > x + R
        G = count_greater_than(x + R)
        if G >= M:
            ans[i] = -1
            continue

        # remove i so we don't count it in the set of potential surpassers
        remove_candidate(i)

        # 2) Let size = # j with A[j] in (x-R, x+R], i.e. strictly > x-R and <= x+R.
        #    That translates to count_in_range(x-R+1, x+R).
        size = count_in_range(x - R + 1, x + R)

        # If G + size < M => i is safe with X=0
        if G + size < M:
            ans[i] = 0
            add_candidate(i)
            continue

        # Otherwise we must find the minimal X.  T = M-1-G
        T = M - 1 - G
        # We want at most T of them to be able to exceed i. That means we want the (T+1)-th largest in that range
        # to fail to exceed i, so we'll find that pivot in ascending.  We'll do k = T+1 from the top => in that range
        # the rank_asc = (size - T). We'll do fenAll's function:
        k = T+1
        # the (k)-th largest from the set S => index = fenAll.find_kth_in_range_desc(l, r, k)
        # We'll do pivot_idx with the partial-range approach:
        # We'll adopt that "find_kth_in_range_desc" function we wrote.  That needs l,r in compressed indices.
        # lC = compress(x-R+1), rC = compress(x+R) or compress(x+R)-1 depending on equality.

        # But let's do it with a simpler logic: we do fenAll.find_kth_in_range_desc(posL, posR, k).
        lval = x - R + 1
        rval = x + R
        # compress them carefully for inclusive range [lval..rval].
        lC = compress(lval)
        # we want <= rval => rC might be compress(rval)
        # if vals[rC] == rval, that's good; otherwise we do rC-1
        rC = compress(rval)
        if rC < len(vals) and vals[rC] == rval:
            pass
        else:
            rC -= 1
        if rC < 0 or lC>rC:
            # no valid pivot => means size =0? Then if G+size >= M => can't fix => -1
            # but we actually know G+size>=M => no solution => -1
            ans[i] = -1
            add_candidate(i)
            continue

        # Now total_in_range = size
        # the (k)-th largest is fenAll.find_kth_in_range_desc(lC, rC, k)
        # but if k> size => that means we want the next pivot below min(S), so effectively
        # we want pivot below the smallest in S => that leads to negative "needed". Actually that means
        # X=0 might already fail. But let's just clamp k <= size. If k>size => then the pivot is below
        # that set => we can treat it as pivot < lval => then needed= maybe 0. But we already concluded
        # if G+size>=M, 0 not good. Let's do the official approach:
        if k> size:
            # Then effectively i needs to exceed every candidate in that range, so the pivot is below
            # the min of that range => that might yield negative needed => interpret that as 0 still not enough.
            # but let's see if i can push X to surpass them all. Actually let's do the formula that the pivot
            # is below the smallest S => pivot < lval => cPivot= (pivot - x -1 + R)//2 => that might be negative
            # => needed=0. But that would conflict with G+size>=M => contradictory. So i must push further if possible.
            # The worst case is that i needs to jump above them, but if k>size that means we are trying to exclude them all
            # from surpassing i, so i must outvote all those size. Possibly we pick the largest A[j] in S and see how many
            # votes i needs to stay above it. Let's just find the largest in S => that is the 1st largest => k=1.
            pivot_idx = fenAll.find_kth_in_range_desc(lC, rC, 1)
        else:
            pivot_idx = fenAll.find_kth_in_range_desc(lC, rC, k)

        if pivot_idx<0 or pivot_idx>=len(vals):
            # out of range => no solution
            ans[i] = -1
            add_candidate(i)
            continue

        pivot_val = vals[pivot_idx]
        # Now compute needed = max(0, floor((pivot_val - x - 1 + R)//2) + 1 ).
        # Because we want A[j] + (R - X) <= x + X for all j up to that pivot,
        # rearranging => 2X >= A[j] + R - x => X >= (A[j] + R - x)/2 => we do integer ceiling => +1 for strict surpass
        # Actually to prevent strict surpass we want A[j] + (R - X) <= x + X => 2X >= A[j] + R - x => X >= (A[j] + R - x + 1)//2 if we want strictly greater
        # but the standard formula is X >= ceil((pivot_val + R - x) / 2). Because "strictly surpass" uses +1, see below:
        # Condition for j not to surpass i is: A[j] + (R - X) <= A[i] + X => A[j] + R - X <= x + X => 2X >= A[j] + R - x.
        # Minimal integer X satifying that is X = ceil((A[j] + R - x)/2). We'll use integer arithmetic carefully:
        # pivot_val => the "k-th largest in S." If i can ensure i's final >= that pivot's worst-case final, i can handle
        # everyone in that set with A <= pivot_val. We'll do the safe side by adding 0.5 in real terms => +1 in integer if needed is fractional:
        #   needed = ( (pivot_val + R - x) + 1 ) // 2.  The code below is the same, just carefully done:
        cPivot = pivot_val + R - x
        # we want X >= cPivot/2, so X >= (cPivot+1)//2 if cPivot≥0, else 0
        if cPivot<=0:
            needed = 0
        else:
            needed = (cPivot + 1)//2

        if needed<0:
            needed = 0
        if needed> R:
            ans[i] = -1
            add_candidate(i)
            continue
        ans[i] = needed

        # re-insert i
        add_candidate(i)

    print(" ".join(map(str, ans)))