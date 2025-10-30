class Solution:
    def maxActiveSectionsAfterTrade(self, s: str) -> int:
        # We will work on the augmented string t = '1' + s + '1'
        # Find all "removable" 1-blocks (surrounded by 0 on both sides in t)
        # and all "flippable" 0-blocks (surrounded by 1 on both sides in t).
        # Initial active count = total ones in s.
        # A trade removes one 1-block of size x and flips one 0-block of size y,
        # net gain = y - x, so final = tot1 + (y - x). Maximize over valid disjoint choices.
        
        n = len(s)
        # build t
        t = '1' + s + '1'
        N = n + 2
        
        # total ones in original s
        tot1 = s.count('1')
        
        # scan t to collect all runs
        runs = []  # (char, start, end) inclusive
        i = 0
        while i < N:
            j = i
            while j < N and t[j] == t[i]:
                j += 1
            runs.append((t[i], i, j-1))
            i = j
        
        # collect removable 1-blocks and flippable 0-blocks
        one_blocks = []  # list of (start, end, size)
        zero_blocks = [] # list of (start, end, size)
        # we can index runs by k
        for k, (ch, a, b) in enumerate(runs):
            if ch == '1':
                # check neighbors in t
                # we know runs[k-1] exists and runs[k+1] exists
                # but need that t[a-1]=='0' and t[b+1]=='0'
                if a-1 >= 0 and b+1 < N and t[a-1] == '0' and t[b+1] == '0':
                    one_blocks.append((a, b, b-a+1))
            else:
                # zero run
                if a-1 >= 0 and b+1 < N and t[a-1] == '1' and t[b+1] == '1':
                    zero_blocks.append((a, b, b-a+1))
        
        # if no removable one-block, no trade possible
        if not one_blocks or not zero_blocks:
            return tot1
        
        # sort one_blocks by start
        one_blocks.sort(key=lambda x: x[0])
        M = len(one_blocks)
        starts = [one_blocks[i][0] for i in range(M)]
        ends   = [one_blocks[i][1] for i in range(M)]
        xs     = [one_blocks[i][2] for i in range(M)]
        
        # build prefix and suffix minima of xs
        pref_min = [0]*M
        suff_min = [0]*M
        pref_min[0] = xs[0]
        for i in range(1, M):
            pref_min[i] = min(pref_min[i-1], xs[i])
        suff_min[M-1] = xs[M-1]
        for i in range(M-2, -1, -1):
            suff_min[i] = min(suff_min[i+1], xs[i])
        
        import bisect
        
        best = tot1  # no-trade option
        # for each zero-block, attempt match
        for (l2, r2, y) in zero_blocks:
            # forbidden region = [l2-1, r2+1]
            forbidden_start = l2-1
            forbidden_end   = r2+1
            # we want removable one-block completely before forbidden_start
            # i.e. end < forbidden_start
            k1 = bisect.bisect_left(ends, forbidden_start) - 1
            cand_x = None
            if k1 >= 0:
                cand_x = pref_min[k1]
            # or completely after forbidden_end: start > forbidden_end
            k2 = bisect.bisect_right(starts, forbidden_end)
            if k2 < M:
                x2 = suff_min[k2]
                cand_x = x2 if cand_x is None else min(cand_x, x2)
            if cand_x is not None:
                cand = tot1 - cand_x + y
                if cand > best:
                    best = cand
        
        return best