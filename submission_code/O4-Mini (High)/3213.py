from typing import List

class Solution:
    def countSubarrays(self, nums: List[int], k: int) -> int:
        n = len(nums)
        # Quick case: any subarray has its max at least once
        if k == 1:
            return n * (n + 1) // 2

        # 1) Compute prev_greater: for each i, index of nearest j<i with nums[j]>nums[i], or -1
        prev_greater = [-1] * n
        stack = []
        for i in range(n):
            while stack and nums[stack[-1]] <= nums[i]:
                stack.pop()
            prev_greater[i] = stack[-1] if stack else -1
            stack.append(i)

        # 2) Compute next_greater: for each i, nearest j>i with nums[j]>nums[i], or n
        next_greater = [n] * n
        stack = []
        for i in range(n - 1, -1, -1):
            while stack and nums[stack[-1]] <= nums[i]:
                stack.pop()
            next_greater[i] = stack[-1] if stack else n
            stack.append(i)

        # 3) Collect positions for each value
        from collections import defaultdict
        positions = defaultdict(list)
        for i, v in enumerate(nums):
            positions[v].append(i)

        total = 0

        # 4) For each value v with freq>=k, build "rectangles" of subarrays that guarantee
        #    at least k occurrences of v and no element >v, then count union area.
        for v, pList in positions.items():
            m = len(pList)
            if m < k:
                continue

            # Build events on l-axis: add or remove an r-interval [c..d]
            # events: (l_pos, type, c, d), type=+1 add, -1 remove
            events = []
            # For each window of k occurrences: pList[j..j+k-1]
            for j in range(0, m - k + 1):
                # left window index j, right window index j+k-1
                left_idx = pList[j]
                right_idx = pList[j + k - 1]
                L = prev_greater[left_idx]    # last >v to left of left_idx
                R = next_greater[right_idx]   # first >v to right of right_idx
                a = L + 1                      # l >= a
                b = left_idx                  # l <= b
                c = right_idx                 # r >= c
                d = R - 1                     # r <= d
                if c > d:
                    # no valid r
                    continue
                # add this rectangle
                events.append((a, +1, c, d))
                # remove at l=b+1
                events.append((b + 1, -1, c, d))

            if not events:
                continue

            # Coordinate-compress the r-endpoints for this v
            coords = []
            for (_, _, c, d) in events:
                coords.append(c)
                coords.append(d + 1)
            coords = sorted(set(coords))
            # map real r to compressed index
            import bisect
            # We'll cover half-open segments [coords[i], coords[i+1])
            seg_n = len(coords) - 1
            # Segment tree arrays
            tree_count = [0] * (4 * seg_n)
            tree_len   = [0] * (4 * seg_n)

            # Build an updater for range [ql..qr] in the compressed segment tree
            def update(node: int, l: int, r: int, ql: int, qr: int, val: int):
                if ql > r or qr < l:
                    return
                if ql <= l and r <= qr:
                    tree_count[node] += val
                else:
                    mid = (l + r) // 2
                    left = node * 2
                    right = node * 2 + 1
                    update(left, l, mid, ql, qr, val)
                    update(right, mid + 1, r, ql, qr, val)
                # After updating counts, fix the covered length
                if tree_count[node] > 0:
                    # fully covered
                    tree_len[node] = coords[r + 1] - coords[l]
                else:
                    if l == r:
                        tree_len[node] = 0
                    else:
                        left = node * 2
                        right = node * 2 + 1
                        tree_len[node] = tree_len[left] + tree_len[right]

            # Sort events by l_pos
            events.sort(key=lambda x: x[0])
            ans_v = 0
            curr_l = events[0][0]
            idx = 0
            E = len(events)
            # Sweep l from smallest event.l to last
            while idx < E:
                l_pos, typ, c, d = events[idx]
                # Accumulate area from curr_l up to l_pos - 1
                if l_pos > curr_l:
                    length_l = l_pos - curr_l
                    covered_r = tree_len[1] if seg_n > 0 else 0
                    ans_v += length_l * covered_r
                    curr_l = l_pos
                # Process all events at l_pos
                while idx < E and events[idx][0] == l_pos:
                    _, t, cc, dd = events[idx]
                    # find compressed indices
                    i_start = bisect.bisect_left(coords, cc)
                    i_end   = bisect.bisect_left(coords, dd + 1) - 1
                    # update [i_start..i_end]
                    if i_start <= i_end:
                        update(1, 0, seg_n - 1, i_start, i_end, t)
                    idx += 1
            # No need to sweep beyond last event: covered_r will be zero afterwards
            total += ans_v

        return total