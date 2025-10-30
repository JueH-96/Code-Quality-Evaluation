class Solution:
    def maxPartitionsAfterOperations(self, s: str, k: int) -> int:
        n = len(s)
        # Quick cases
        if n == 0:
            return 0
        # If k >= 26, no break ever (or trivial), and we cannot improve beyond 1
        # Also if total distinct in s <= k, no break ever.
        if k >= 26 or len(set(s)) <= k:
            return 1

        # Convert s to indices 0..25
        s_idx = [ord(c) - 97 for c in s]

        # Precompute next occurrence positions: next_occ[i][c] = smallest j >= i with s[j]==c, else n
        INF = n
        # initialize last_occ to INF
        last_occ = [INF] * 26
        # allocate next_occ
        next_occ = [None] * (n + 1)
        next_occ[n] = list(last_occ)
        # fill backwards
        for i in range(n - 1, -1, -1):
            ci = s_idx[i]
            last_occ[ci] = i
            # shallow copy
            next_occ[i] = list(last_occ)

        # DP g[i] = number of segments for suffix s[i:] by greedy with at most k distinct
        # segmentation resets at i
        g = [0] * (n + 1)
        g[n] = 0
        # temp list for occ
        occ_tmp = [0] * 26
        for i in range(n - 1, -1, -1):
            # build list of next occurrences
            lo = next_occ[i]
            # copy to temp
            # We need occ_list sorted
            for c in range(26):
                occ_tmp[c] = lo[c]
            # sort
            # occ_list = sorted(occ_tmp)
            # We only need (k)-th index => occ_list[k]
            # But sorting 26 is cheap
            occ_tmp.sort()
            # breakpos = occ_list[k]
            # If breakpos < n, segment 1 + g[breakpos], else 1
            bp = occ_tmp[k]  # k-th index, 0-based
            if bp < n:
                g[i] = 1 + g[bp]
            else:
                g[i] = 1

        # Now simulate original segmentation to get seg_idx for each position and seg_infos
        seg_idx = [0] * n
        seg_infos = []  # list of dicts: each has start, posk, end, mask
        curr_seg = 0
        D_mask = 0
        D_count = 0
        posk = None
        l = 0
        for i in range(n):
            ci = s_idx[i]
            # Check if break before including s[i]
            if ((D_mask >> ci) & 1) == 0 and D_count == k:
                # finish current segment
                seg_infos.append({
                    'start': l,
                    'posk': posk,
                    'end': i,
                    'mask': D_mask
                })
                # start new segment at i
                curr_seg += 1
                l = i
                D_mask = 0
                D_count = 0
                posk = None
            # include s[i] in current segment
            if ((D_mask >> ci) & 1) == 0:
                D_mask |= (1 << ci)
                D_count += 1
                # record posk when D_count hits k
                if D_count == k:
                    posk = i
            # record segment index for this position
            seg_idx[i] = curr_seg
        # finish last segment
        seg_infos.append({
            'start': l,
            'posk': posk,
            'end': n,
            'mask': D_mask
        })

        # Original segmentation count
        orig_segments = g[0]

        best = orig_segments

        # Prepare a reusable array for sorting (for c,new occs)
        # Now consider each full segment with posk and possible p = posk+1 < end
        # Candidate p list of tuples (p, seg_no, mask)
        cand = []
        # segments are indexed 0..len(seg_infos)-1
        for t, info in enumerate(seg_infos):
            pk = info['posk']
            if pk is not None:
                p = pk + 1
                if p < info['end']:
                    # candidate change position
                    cand.append((p, t, info['mask']))

        # For each candidate p, compute best f(s') by trying all c_new not in mask
        # We'll use next_occ[p+1] list and sort pairs (occ, c) per p
        for p, t, seg_mask in cand:
            # Build sorted list A of (occ, c) for c in 0..25 by next_occ[p+1][c]
            # If p+1==n, next_occ[n] all INF
            lo = next_occ[p + 1] if p + 1 <= n else next_occ[n]
            # build pairs
            A = [(lo[c], c) for c in range(26)]
            A.sort(key=lambda x: x[0])
            # build idx_in_A
            idx_in_A = [0] * 26
            for j, (_, c) in enumerate(A):
                idx_in_A[c] = j
            # for this p, best g_mod
            best_gmod = 1  # at least one segment
            # try all c_new not in seg_mask
            # seg_mask bits of letters already in D_old_set
            # only letters not in D_mask can cause break at p
            # but if we choose c_new in mask, break won't occur at p, which is not beneficial
            # so skip c in mask
            for c_new in range(26):
                if (seg_mask >> c_new) & 1:
                    continue
                idx_c = idx_in_A[c_new]
                # derive j for breakpos selection
                # B = A without element at idx_c; need B[k-1]
                # if idx_c > k-1 then B[k-1] = A[k-1] else B[k-1] = A[k]
                if idx_c > k - 1:
                    j = k - 1
                else:
                    j = k
                # get occ at A[j]
                if j < 26:
                    occj, _ = A[j]
                else:
                    occj = INF
                if occj < n:
                    # break occurs at occj
                    gmod = 1 + g[occj]
                else:
                    # no break in suffix
                    gmod = 1
                # track best
                if gmod > best_gmod:
                    best_gmod = gmod
            # total segments = segments before p + segments in suffix
            # segments before p = seg_idx[p] + 1
            total = (seg_idx[p] + 1) + best_gmod
            if total > best:
                best = total

        return best