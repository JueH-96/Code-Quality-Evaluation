class Solution:
    def maximumCoins(self, coins: List[List[int]], k: int) -> int:
        # Build events for the piecewise-constant weight function w(x)
        events = {}
        for l, r, c in coins:
            events[l] = events.get(l, 0) + c
            events[r + 1] = events.get(r + 1, 0) - c

        # Sort event positions
        keys = sorted(events)
        # Build intervals of constant weight: arr_w[i], arr_len[i]
        arr_w = []
        arr_len = []
        cur_w = 0
        for i, pos in enumerate(keys):
            cur_w += events[pos]
            if i + 1 < len(keys):
                nxt = keys[i + 1]
                length = nxt - pos
                if length > 0:
                    arr_w.append(cur_w)
                    arr_len.append(length)

        # Prepend an initial zero‐weight interval of length k
        # This covers windows that start before the first coin segment
        arr_w.insert(0, 0)
        arr_len.insert(0, k)

        # Two‐pointer sweep to find max sum over any subinterval of length exactly k
        l = 0
        cur_sum = 0
        cur_len = 0
        ans = 0
        M = len(arr_w)

        for r in range(M):
            # Add the r‐th interval fully
            w_r = arr_w[r]
            len_r = arr_len[r]
            cur_sum += w_r * len_r
            cur_len += len_r

            # While we can drop an entire left interval and still keep length >= k, do so
            while l <= r and cur_len - arr_len[l] >= k:
                cur_sum -= arr_w[l] * arr_len[l]
                cur_len -= arr_len[l]
                l += 1

            # If we have at least k length, compute the exact window sum by
            # removing the excess (cur_len - k) from the leftmost interval
            if cur_len >= k:
                excess = cur_len - k
                # we only ever remove from arr_w[l], because we stopped dropping
                # full intervals once dropping more would make length < k
                window_sum = cur_sum - excess * arr_w[l]
                if window_sum > ans:
                    ans = window_sum

        return ans