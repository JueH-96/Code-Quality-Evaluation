from typing import List

class Solution:
    def minimumValueSum(self, nums: List[int], andValues: List[int]) -> int:
        n = len(nums)
        m = len(andValues)
        # A large sentinel for "infinity"
        INF = 10**18

        # Precompute, for each end index r, the distinct AND-values of subarrays ending at r,
        # along with the minimal start index that achieves each AND-value.
        segments_by_end = []
        prev_list = []  # list of (and_val, start) for r-1
        for r in range(n):
            curr_map = {nums[r]: r}
            # extend all previous AND-blocks by nums[r]
            for val, start in prev_list:
                new_val = val & nums[r]
                # keep the minimal start for each new_val
                if new_val in curr_map:
                    if start < curr_map[new_val]:
                        curr_map[new_val] = start
                else:
                    curr_map[new_val] = start
            # build the new list sorted by start ascending
            items = list(curr_map.items())  # (and_val, start)
            items.sort(key=lambda x: x[1])
            prev_list = items
            segments_by_end.append(items)

        # Precompute logs for RMQ
        # logs[i] = floor(log2(i))
        N = n + 1
        logs = [0] * (N + 1)
        for i in range(2, N + 1):
            logs[i] = logs[i // 2] + 1

        # dp_prev[i] = minimal sum of last-elements for partitioning first i elements
        # into (p-1) segments, for the current p.
        dp_prev = [INF] * (n + 1)
        dp_prev[0] = 0

        # Iterate over segments 1..m
        for p in range(1, m + 1):
            target = andValues[p - 1]
            # Build a sparse table on dp_prev for range-min queries
            K = logs[N]
            st = [None] * (K + 1)
            st[0] = dp_prev
            length = N
            for k in range(1, K + 1):
                prev_row = st[k - 1]
                step = 1 << (k - 1)
                new_len = length - (1 << k) + 1
                row = [0] * new_len
                for i in range(new_len):
                    a = prev_row[i]
                    b = prev_row[i + step]
                    row[i] = a if a < b else b
                st[k] = row

            # dp_curr for p segments
            dp_curr = [INF] * (n + 1)

            # Fill dp_curr by trying to end the p-th segment at each r (0-based)
            for r in range(n):
                i = r + 1
                # need at least p elements to make p segments
                if i < p:
                    continue
                segs = segments_by_end[r]
                # find the block where AND == target
                # segs: list of (and_val, start) sorted by start
                for idx, (val, L) in enumerate(segs):
                    if val == target:
                        # compute R: the maximal start for this block
                        if idx + 1 < len(segs):
                            R = segs[idx + 1][1] - 1
                        else:
                            R = r
                        # range-min query on dp_prev[L..R]
                        length = R - L + 1
                        k = logs[length]
                        row = st[k]
                        # candidate minimum in that range
                        # two overlapping intervals technique
                        right_start = R - (1 << k) + 1
                        min_dp = row[L] if row[L] < row[right_start] else row[right_start]
                        if min_dp < INF:
                            cost = min_dp + nums[r]
                            if cost < dp_curr[i]:
                                dp_curr[i] = cost
                        break

            dp_prev = dp_curr

        ans = dp_prev[n]
        return ans if ans < INF else -1