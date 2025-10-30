from bisect import bisect_left, bisect_right, insort
from typing import List

class Solution:
    def numberOfAlternatingGroups(self, colors: List[int], queries: List[List[int]]) -> List[int]:
        n = len(colors)
        # diff[i] = 1 if colors[i] != colors[(i+1)%n]
        diff = [0] * n
        for i in range(n):
            diff[i] = 1 if colors[i] != colors[(i+1) % n] else 0

        # We'll maintain segments of consecutive 1s in diff (circular)
        # track: starts = sorted list of start indices of segments
        # lengths: dict start->length
        starts = []
        lengths = {}
        # size_count: map from group size (length+1) to count
        size_count = {}

        def inc_size(sz, delta):
            size_count[sz] = size_count.get(sz, 0) + delta
            if size_count[sz] == 0:
                del size_count[sz]

        # initialize segments
        visited = [False]*n
        for i in range(n):
            if diff[i] == 1 and not visited[i]:
                # walk the segment
                j = i
                length = 0
                while diff[j] == 1 and not visited[j]:
                    visited[j] = True
                    length += 1
                    j = (j + 1) % n
                start = i
                # record
                starts.append(start)
                lengths[start] = length
                inc_size(length+1, 1)
        starts.sort()

        def find_segment(p):
            """Return (start, length) of segment covering diff-index p, or (None,0)."""
            if not starts:
                return (None, 0)
            # binary search
            idx = bisect_right(starts, p) - 1
            if idx >= 0:
                s = starts[idx]
                L = lengths[s]
                end = s + L - 1
                # no wrap
                if end < n:
                    if s <= p <= end:
                        return (s, L)
                else:
                    # wrap case end>=n
                    end_mod = end % n
                    if p >= s or p <= end_mod:
                        return (s, L)
            else:
                # p < first start, check last wrap
                s = starts[-1]
                L = lengths[s]
                end = s + L - 1
                if end >= n:
                    end_mod = end % n
                    if p <= end_mod:
                        return (s, L)
            return (None, 0)

        def remove_segment(s):
            """Remove segment at start s."""
            L = lengths.pop(s)
            # remove s from starts
            idx = bisect_left(starts, s)
            starts.pop(idx)
            inc_size(L+1, -1)
            return L

        def add_segment(s, L):
            """Add segment starting at s of length L."""
            insort(starts, s)
            lengths[s] = L
            inc_size(L+1, 1)

        res = []

        for q in queries:
            if q[0] == 1:
                sz = q[1]
                res.append(size_count.get(sz, 0))
            else:
                # update color at idx
                _, idx, c = q
                if colors[idx] == c:
                    continue
                # two diffs to maybe update: at pos idx-1 and idx
                for dpos in [(idx-1) % n, idx]:
                    old = diff[dpos]
                    new = 1 if colors[dpos] != colors[(dpos+1)%n] else 0
                    if old == new:
                        continue
                    diff[dpos] = new
                    if new == 1:
                        # turn 0->1, merge neighbors
                        # left seg if covers dpos-1
                        left_pos = (dpos-1) % n
                        sL, L = (None,0)
                        if diff[left_pos] == 1:
                            sL, L = find_segment(left_pos)
                            remove_segment(sL)
                        # right seg if covers dpos+1
                        right_pos = (dpos+1) % n
                        sR, R = (None,0)
                        if diff[dpos+1 if dpos+1<n else 0] == 1:
                            sR, R = find_segment(right_pos)
                            remove_segment(sR)
                        # new start is sL if exists else dpos
                        if sL is not None:
                            new_start = sL
                        else:
                            new_start = dpos
                        new_len = L + 1 + R
                        add_segment(new_start, new_len)
                    else:
                        # turn 1->0, split segment that covers dpos
                        s, L = find_segment(dpos)
                        if s is None:
                            continue
                        oldL = remove_segment(s)
                        # segment spanned positions s..s+L-1 mod n
                        # compute offset of dpos in this segment
                        # normalized: compute linear positions
                        # if no wrap:
                        end = s + oldL - 1
                        if end < n:
                            offset = dpos - s
                        else:
                            # wrap: treat p>=s as p-s, else p+ n - s
                            if dpos >= s:
                                offset = dpos - s
                            else:
                                offset = dpos + n - s
                        # left piece length = offset
                        if offset > 0:
                            add_segment(s, offset)
                        # right piece length = oldL - offset -1
                        right_len = oldL - offset - 1
                        if right_len > 0:
                            new_start = (s + offset + 1) % n
                            add_segment(new_start, right_len)
                colors[idx] = c

        return res