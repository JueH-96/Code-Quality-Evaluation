class Solution:
    def numberOfAlternatingGroups(self, colors: list, queries: list) -> list:
        n = len(colors)
        # Build diff array: diff[i] = 1 if tile i and (i+1)%n have different colors.
        diff = [0] * n
        for i in range(n):
            diff[i] = 1 if colors[i] != colors[(i+1) % n] else 0
        ones_count_diff = sum(diff)
        
        #---------------------------------------------------------------------
        # We will “segment” the cyclic diff array into maximal blocks (segments) of ones.
        # A segment will be represented as a tuple (l, r) meaning that diff[l], diff[l+1],...,diff[r] 
        # (modulo n) are all 1. (If l<=r the segment is “linear” – if l>r it wraps around.)
        segs = []
        i = 0
        while i < n:
            if diff[i] == 1:
                start = i
                while i < n and diff[i] == 1:
                    i += 1
                end = i - 1
                segs.append((start, end))
            else:
                i += 1
        # In a cyclic array if both diff[0] and diff[n–1] are 1 then the first and last segments merge.
        if segs and diff[0] == 1 and diff[n-1] == 1 and len(segs) > 1:
            first = segs[0]
            last = segs[-1]
            merged = (last[0], first[1])
            segs = segs[1:-1]
            segs.append(merged)
            segs.sort(key=lambda x: x[0])
        #---------------------------------------------------------------------
        # Fenw–tree structure (1-indexed) to maintain frequency information.
        class Fenw:
            def __init__(self, m):
                self.n = m
                self.f = [0]*(m+1)
            def update(self, i, delta):
                while i <= self.n:
                    self.f[i] += delta
                    i += i & -i
            def query(self, i):
                s = 0
                while i:
                    s += self.f[i]
                    i -= i & -i
                return s
        
        # In our fenw trees we support lengths 1..n.
        fenwCount = Fenw(n)
        fenwSum = Fenw(n)
        
        # Helper: segment length (works for both linear and wrap-around segments)
        def seg_length(seg):
            l, r = seg
            if l <= r:
                return r - l + 1
            else:
                return (n - l) + (r + 1)
        
        # Initially “add” every segment to our two Fenw–trees.
        for seg in segs:
            L = seg_length(seg)
            if L > 0:
                fenwCount.update(L, 1)
                fenwSum.update(L, L+1)
                
        #---------------------------------------------------------------------
        # We now need to update our segmentation structure when a diff entry changes.
        # (The tricky part is that our diff array is cyclic.)
        #
        # Helper: given a segment seg = (l, r) and an index x, check if x is contained in seg.
        def seg_contains(seg, x):
            l, r = seg
            if l <= r:
                return l <= x <= r
            else:
                return x >= l or x <= r

        # Using bisect we “locate” the segment (if any) covering a given index x.
        import bisect
        def find_seg(x):
            if not segs:
                return None
            i = bisect.bisect_right(segs, (x, float('inf')))
            if i:
                seg = segs[i-1]
                if seg[0] <= seg[1]:
                    if seg[0] <= x <= seg[1]:
                        return i-1
                else:
                    if x >= seg[0] or x <= seg[1]:
                        return i-1
            # (One more check in case a wrap–around segment is at index 0.)
            seg = segs[0]
            if seg[0] > seg[1]:
                if x >= seg[0] or x <= seg[1]:
                    return 0
            return None
        
        # When a segment is removed (deleted) we subtract its contribution from Fenw.
        def remove_seg_by_index(idx):
            seg = segs.pop(idx)
            L = seg_length(seg)
            fenwCount.update(L, -1)
            fenwSum.update(L, -(L+1))
            return seg
        
        # To “insert” a segment we add it in sorted order (by its starting index) and update Fenw.
        def add_seg(seg):
            L = seg_length(seg)
            bisect.insort(segs, seg)
            fenwCount.update(L, 1)
            fenwSum.update(L, L+1)
        
        # For a segment from which we are removing an index x, we “split” the segment.
        def split_seg(seg, x):
            l, r = seg
            res = []
            if l <= r:
                if l == r:
                    return res
                if x == l:
                    if l+1 <= r:
                        res.append((l+1, r))
                elif x == r:
                    if l <= r-1:
                        res.append((l, r-1))
                else:
                    if l <= x-1:
                        res.append((l, x-1))
                    if x+1 <= r:
                        res.append((x+1, r))
            else:
                # In a wrap–around segment, we use a linearization trick.
                # Let L_lin = l and R_lin = r+n so that seg covers indices L_lin,...,R_lin.
                L_lin = l
                R_lin = r + n
                x_lin = x if x >= l else x + n
                if x_lin == L_lin:
                    new_linear = [(L_lin+1, R_lin)]
                elif x_lin == R_lin:
                    new_linear = [(L_lin, R_lin-1)]
                else:
                    new_linear = []
                    if L_lin <= x_lin-1:
                        new_linear.append((L_lin, x_lin-1))
                    if x_lin+1 <= R_lin:
                        new_linear.append((x_lin+1, R_lin))
                # Convert each linear interval [a, b] back to a circular segment.
                for (a, b) in new_linear:
                    if b < n:
                        res.append((a, b))
                    else:
                        res.append((a, b-n))
            return res
        
        # When diff[i] changes we update our segmentation structure appropriately.
        # (Recall that an update in the original colors changes two diff entries.)
        def update_diff(i, new_val):
            nonlocal ones_count_diff
            old = diff[i]
            if old == new_val:
                return
            diff[i] = new_val
            if new_val == 1:
                ones_count_diff += 1
                # We are “inserting” a 1 at position i. Check our neighbours:
                merge_left = (diff[(i-1) % n] == 1)
                merge_right = (diff[(i+1) % n] == 1)
                if not merge_left and not merge_right:
                    add_seg((i, i))
                elif merge_left and not merge_right:
                    idx = find_seg((i-1) % n)
                    if idx is not None:
                        seg = remove_seg_by_index(idx)
                        new_seg = (seg[0], i)
                        add_seg(new_seg)
                    else:
                        add_seg((i, i))
                elif not merge_left and merge_right:
                    idx = find_seg((i+1) % n)
                    if idx is not None:
                        seg = remove_seg_by_index(idx)
                        new_seg = (i, seg[1])
                        add_seg(new_seg)
                    else:
                        add_seg((i, i))
                else:
                    idx_left = find_seg((i-1) % n)
                    idx_right = find_seg((i+1) % n)
                    seg_left = None
                    seg_right = None
                    if idx_left is not None:
                        seg_left = segs[idx_left]
                    if idx_right is not None:
                        # It is possible that after removal the list changes; so re‐find.
                        new_idx = find_seg((i+1) % n)
                        if new_idx is not None:
                            idx_right = new_idx
                            seg_right = segs[idx_right]
                    if seg_left is not None and seg_right is not None and idx_left != idx_right:
                        # Remove the two segments and merge them.
                        if idx_left > idx_right:
                            seg_left = remove_seg_by_index(idx_left)
                            seg_right = remove_seg_by_index(idx_right)
                        else:
                            seg_right = remove_seg_by_index(idx_right)
                            seg_left = remove_seg_by_index(idx_left)
                        new_seg = (seg_left[0], seg_right[1])
                        add_seg(new_seg)
                    elif seg_left is not None:
                        seg_left = remove_seg_by_index(idx_left)
                        new_seg = (seg_left[0], i)
                        add_seg(new_seg)
                    elif seg_right is not None:
                        seg_right = remove_seg_by_index(idx_right)
                        new_seg = (i, seg_right[1])
                        add_seg(new_seg)
                    else:
                        add_seg((i, i))
            else:
                ones_count_diff -= 1
                # We are removing a 1 from diff[i]. It must lie in some segment.
                idx = find_seg(i)
                if idx is None:
                    return
                seg = remove_seg_by_index(idx)
                new_segs = split_seg(seg, i)
                for ns in new_segs:
                    add_seg(ns)
        
        #---------------------------------------------------------------------
        # Now process the queries. There are two types:
        #   Type 1: [1, size] must answer the number of alternating groups (of that many tiles)
        #   Type 2: [2, index, color] updates colors at index.
        #
        # For a given “group size” k, let w = k–1. Then a group starting at i is alternating if 
        # the contiguous block (in diff) of ones of length L starting at i is at least w.
        # Over every segment of diff (of ones) of length L (if L ≥ w) there are (L–w+1) valid starting positions.
        # (Our Fenw–trees maintain the frequencies of segment–lengths so that the total answer is
        #   sum₍L ≥ w₎ (L–w+1)  =  (sum₍L ≥ w₎ (L+1)) – w*(# segments with L ≥ w).
        # Special–case: if ones_count_diff == n then every starting index gives an alternating group so answer = n.
        
        res_ans = []
        for q in queries:
            if q[0] == 1:
                k = q[1]
                w = k - 1
                if ones_count_diff == n:
                    res_ans.append(n)
                else:
                    cnt = fenwCount.query(n) - fenwCount.query(w-1)
                    ssum = fenwSum.query(n) - fenwSum.query(w-1)
                    ans = ssum - w * cnt
                    res_ans.append(ans)
            else:
                # Query type 2: [2, index, color] – update colors.
                index, new_color = q[1], q[2]
                if colors[index] == new_color:
                    continue
                colors[index] = new_color
                # Only two diff entries are affected:
                pos1 = (index - 1) % n
                pos2 = index
                new_val1 = 1 if colors[pos1] != colors[index] else 0
                new_val2 = 1 if colors[index] != colors[(index+1) % n] else 0
                update_diff(pos1, new_val1)
                update_diff(pos2, new_val2)
        return res_ans