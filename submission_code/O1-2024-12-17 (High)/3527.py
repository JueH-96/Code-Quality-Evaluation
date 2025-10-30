class Solution:
    def numberOfAlternatingGroups(self, colors: List[int], queries: List[List[int]]) -> List[int]:
        """
        We maintain a "diff" array of length N (circular):
            diff[i] = 1  if  colors[i] != colors[(i+1) mod N]
                     0  otherwise
        An alternating group of size K in the original circle
        corresponds to a contiguous block of length K-1 in diff
        (modulo N) that is all 1s. We need to count how many
        starting positions i (0 <= i < N) yield K-1 consecutive 1s.

        Instead of counting directly for each query, we maintain
        a decomposition of diff into runs of consecutive 1s,
        separated by positions of 0. The circle is split at each 0
        in diff. Between consecutive zeros (z1, z2), there is a run
        of length runLen = (z2 - z1 - 1) mod N of 1s.

        We store how many runs of length r exist (freq[r]), and
        answer queries using a prefix-sum / Fenwick-tree trick:
          - If the entire diff is 1s, then freq[N] == 1, and
            any subarray of length K-1 is valid from each of
            the N starting points (answer = N).
          - Otherwise, we sum over all runs:
                number of sub-subarrays of length L = (K-1)
                in a run of length r is max(0, r - L + 1).
            We precompute partial sums of freq[r] and r*freq[r]
            in Fenwicks so that a query can be answered quickly.

        When diff[x] flips from 0->1 or 1->0 (due to a color change),
        we update the set of zeros and the freq array accordingly.
        - 0->1 removes a zero from our set (merging two runs).
        - 1->0 adds a zero (splitting one run into two).

        Each query of type [1, size] returns how many
        alternating groups of that size exist.
        Each query of type [2, idx, newColor] adjusts colors[idx]
        and updates diff accordingly.

        The data structure operations each take O(log N), and
        we can handle up to 5e4 queries within time limits if
        implemented efficiently.
        """

        import sys
        import bisect

        input = sys.stdin.readline  # Not strictly needed in this environment, but good practice.

        # Fenwick (Binary Indexed) Tree for range-sum queries.
        # We'll keep two Fenwicks:
        #   fenwicksFreq[i] = sum of freq[1..i]
        #   fenwicksFreqR[i] = sum of (r * freq[r]) for r in [1..i]
        # This allows us to compute:
        #   sum_{r=L..R} freq[r]   = prefixFreq(R) - prefixFreq(L-1)
        #   sum_{r=L..R} r*freq[r] = prefixFreqR(R) - prefixFreqR(L-1)
        # in O(log N).
        class Fenwick:
            def __init__(self, n):
                self.n = n
                self.fw = [0]*(n+1)  # 1-based indexing

            def update(self, i, delta):
                # i in [1..n]
                while i <= self.n:
                    self.fw[i] += delta
                    i += i & -i

            def query(self, i):
                # sum from 1..i
                s = 0
                while i > 0:
                    s += self.fw[i]
                    i -= i & -i
                return s

            def range_query(self, l, r):
                # sum from l..r
                if r < l:
                    return 0
                return self.query(r) - self.query(l-1)

        # Build Fenwicks for freq[r] and r*freq[r].
        # freq array covers r in [0..N], though we only really
        # use [1..N]. freq[N] == 1 means "all ones".
        # We'll do 1-based Fenwicks: index r goes to r in Fenwicks.
        # So fenwicksFreq.update(r, +1) is how we increment freq[r].
        # fenwicksFreqR.update(r, +r) is how we increment r*freq[r].
        n = len(colors)
        fenwicksFreq = Fenwick(n)      # for freq[r]
        fenwicksFreqR = Fenwick(n)     # for r*freq[r]
        
        # freq[r] = how many runs of length r in diff
        freq = [0]*(n+1)
        
        # A helper to do "freq[r] += d" and update Fenwicks.
        def add_freq(r, d):
            # r in [0..n], but we only store Fenwicks for [1..n].
            # If r==0, it's a run of length 0 -> doesn't count for queries, skip.
            if r == 0:
                return
            freq[r] += d
            fenwicksFreq.update(r, d)
            fenwicksFreqR.update(r, d*r)

        # Returns the length of the run of 1s between two zeros
        # z1, z2 in the circular sense (z2 is the "next zero" after z1).
        def run_length(z1, z2):
            if z2 > z1:
                return (z2 - z1 - 1)
            else:
                # wrap around
                return (z2 + n) - z1 - 1

        # We'll keep track of the positions of diff that are zero in a
        # sorted list zeroPos. Then each pair of consecutive zeros
        # in that list yields a run of 1s. If zeroPos is empty,
        # that means diff is all 1s -> freq[n] = 1.
        zeroPos = []

        # Build the diff array.
        diff = [0]*n
        for i in range(n):
            if colors[i] != colors[(i+1) % n]:
                diff[i] = 1
            else:
                diff[i] = 0

        # Initialize zeroPos from diff.
        for i in range(n):
            if diff[i] == 0:
                zeroPos.append(i)

        zeroPos.sort()
        M = len(zeroPos)  # number of zeroes in diff

        # allOnes indicates whether diff is entirely ones.
        allOnes = False
        if M == 0:
            # entire diff is 1 => one big run of length n
            freq[n] = 1
            fenwicksFreq.update(n, 1)
            fenwicksFreqR.update(n, n)
            allOnes = True
        else:
            # Build runs from consecutive zeros in zeroPos
            for i in range(M):
                z1 = zeroPos[i]
                z2 = zeroPos[(i+1) % M]
                length_run = run_length(z1, z2)
                add_freq(length_run, 1)

        # Functions to handle updates in diff (flip 0->1 or 1->0).
        # "addOne(x)" = diff[x] changes from 0->1 => remove x from zeroPos => merge runs.
        # "removeOne(x)" = diff[x] changes from 1->0 => add x to zeroPos => split runs.
        def addOne(x):
            # oldVal=0-> newVal=1
            # If we were "allOnes", then diff[x] wouldn't be 0.
            # so if allOnes is True, no change actually can happen here.
            nonlocal M, allOnes
            if allOnes:
                # In a truly all-ones scenario, we can't flip 0->1
                # because there are no zeros. So nothing to do.
                return

            # Remove x from zeroPos
            pos = bisect.bisect_left(zeroPos, x)
            if pos == len(zeroPos) or zeroPos[pos] != x:
                # Should not happen if oldVal=0. Just in case.
                return

            # If M==1 and we remove x => zeroPos becomes empty => all ones
            if M == 1:
                # we had 1 zero => a single run => remove that => now all ones
                # That single run was of length (n-1).
                # freq array had freq[n-1] or something, or maybe more complicated if
                # ring structure. But simpler to just do:
                # remove the run of length run_length(x,x) => that is n-1
                length_run = run_length(x, x)  # = n-1
                add_freq(length_run, -1)
                # Now zeroPos = []
                zeroPos.pop(pos)
                M = 0
                # Become all ones
                allOnes = True
                add_freq(n, +1)
                return

            # We have at least 2 zeros. We'll unify the runs that used to go [z_p -> x], [x -> z_n]
            # into a single run [z_p -> z_n].
            # find z_p = zeroPos[pos-1], z_n = zeroPos[(pos+1) % M]
            # but we must handle out-of-range for pos-1
            z_p = zeroPos[pos-1] if pos-1 >= 0 else zeroPos[M-1]
            z_n = zeroPos[(pos+1) % M]

            old_run1 = run_length(z_p, x)
            old_run2 = run_length(x, z_n)
            new_run  = run_length(z_p, z_n)

            add_freq(old_run1, -1)
            add_freq(old_run2, -1)
            add_freq(new_run, +1)

            # remove x from zeroPos
            zeroPos.pop(pos)
            M -= 1

        def removeOne(x):
            # oldVal=1 -> newVal=0
            # If we are allOnes => freq[n] = 1 => we break it now.
            nonlocal M, allOnes
            if allOnes:
                # remove the big run n => freq[n]--
                add_freq(n, -1)
                freq[n] = 0
                allOnes = False
                # Now we have exactly 1 zero => that yields one run of length n-1
                zeroPos.append(x)
                zeroPos.sort()
                M = 1
                add_freq(n-1, +1)
                return

            # We insert x into zeroPos, splitting a run [z_p->z_n] into [z_p->x] + [x->z_n].
            pos = bisect.bisect_left(zeroPos, x)
            if pos == len(zeroPos):
                # x is beyond all existing zeroPos, so we wrap to index 0 for next
                z_n = zeroPos[0]
                z_p = zeroPos[-1]
                # but we will put x at the end
            else:
                # the "next" zero in sorted order is zeroPos[pos]
                z_n = zeroPos[pos]
                # the previous zero is zeroPos[pos-1], if pos > 0, else zeroPos[-1]
            z_p = zeroPos[pos-1] if pos > 0 else zeroPos[-1]

            # The run to remove is the run [z_p->z_n]
            old_run = run_length(z_p, z_n)
            add_freq(old_run, -1)

            # We add two runs: [z_p->x], [x->z_n]
            run1 = run_length(z_p, x)
            run2 = run_length(x, z_n)
            add_freq(run1, +1)
            add_freq(run2, +1)

            # Insert x into zeroPos
            zeroPos.insert(pos, x)
            M += 1

        # Recompute diff[i-1], diff[i] after a color change at index i
        # and update the data structure as needed.
        def updateColor(i, newC):
            oldC = colors[i]
            if oldC == newC:
                return
            colors[i] = newC
            # Two diff positions can change:
            #   diff[i-1], comparing colors[i-1], colors[i]
            #   diff[i],   comparing colors[i], colors[i+1]
            i_left = (i - 1) % n
            i_right = i

            old_d_left = diff[i_left]
            new_d_left = 1 if (colors[i_left] != colors[i]) else 0
            if old_d_left != new_d_left:
                diff[i_left] = new_d_left
                if old_d_left == 0 and new_d_left == 1:
                    # addOne(i_left)
                    addOne(i_left)
                else:
                    # removeOne(i_left)
                    removeOne(i_left)

            old_d_right = diff[i_right]
            new_d_right = 1 if (colors[i_right] != colors[(i_right+1) % n]) else 0
            if old_d_right != new_d_right:
                diff[i_right] = new_d_right
                if old_d_right == 0 and new_d_right == 1:
                    addOne(i_right)
                else:
                    removeOne(i_right)

        # Query function: how many alternating groups of size K?
        #   Let L = K-1. If allOnes => answer = n (since any start 0..n-1 is valid).
        #   Otherwise we sum_{r=L..n-1} freq[r] * (r - L + 1).
        #   We do partial sums in the Fenwicks.

        results = []

        def getQuery(K):
            L = K - 1
            if allOnes:
                # If the whole diff is 1 -> from each of n starting points
                # we have a valid subarray of length L, provided L <= n.
                # Here K <= n-1 => L <= n-1, so valid => answer = n
                return n

            # If L > n-1, then 0. But the problem states K <= n-1 => L <= n-2 in normal usage.
            if L > n-1:
                return 0

            # sum_{r=L..n-1} freq[r] * (r - L + 1)
            #   = sum_{r=L..n-1} [ r*freq[r] ] - (L-1)* sum_{r=L..n-1} freq[r]
            # Actually (r - L + 1) = (r + 1 - L).
            # We'll do it with Fenwicks: let SR(r) = sum_{i=1..r} i*freq[i], SF(r) = sum_{i=1..r} freq[i].
            # Sub-sum from L..(n-1):
            #   sumR = SR(n-1) - SR(L-1)
            #   sumF = SF(n-1) - SF(L-1)
            # Then total = sum_{r=L..n-1} [r - L + 1]* freq[r]
            #            = sum_{r=L..n-1} r*freq[r] - (L - 1)* sum_{r=L..n-1} freq[r]
            # We can do that as:
            #   sumR - (L-1)* sumF
            Rmax = n-1  # top of range
            Lmin = L    # bottom of range
            if Lmin < 1:
                Lmin = 1
            if Lmin > Rmax:
                return 0

            sum_r = fenwicksFreqR.range_query(Lmin, Rmax)
            sum_f = fenwicksFreq.range_query(Lmin, Rmax)
            # result
            ans = sum_r - (L-1)*sum_f
            return ans if ans > 0 else 0

        # Process queries
        for q in queries:
            if q[0] == 1:
                # [1, size_i] => ask how many groups of size size_i
                K = q[1]
                ans = getQuery(K)
                results.append(ans)
            else:
                # [2, index_i, color_i] => update
                idx, newC = q[1], q[2]
                updateColor(idx, newC)

        return results