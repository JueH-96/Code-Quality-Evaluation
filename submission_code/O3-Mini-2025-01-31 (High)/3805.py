class Solution:
    def maxActiveSectionsAfterTrade(self, s: str) -> int:
        # baseline: count of active sections in the original string
        baseline = s.count('1')
        n = len(s)
        # If there's no possibility of any trade improvement, return baseline.
        # For example, if s is all 0's or all 1's there is no valid removal.
        if n == 0:
            return 0

        # Build the augmented string: add a '1' at both ends.
        t = "1" + s + "1"
        m = len(t)  # m = n+2

        # Build run-length encoding of t.
        # Each element will be a tuple (char, count)
        runs = []
        i = 0
        while i < m:
            ch = t[i]
            cnt = 1
            j = i + 1
            while j < m and t[j] == ch:
                cnt += 1
                j += 1
            runs.append((ch, cnt))
            i = j

        # The trade operation consists of two flips:
        # 1. Remove a contiguous block of '1's that is "surrounded by" 0's.
        #    In the augmented string, that means the candidate '1'-run must not be
        #    the first or last run (because the augmented boundaries are '1').
        # 2. Then, flip the whole contiguous zero block surrounding that removed block.
        #
        # After removal, the zeros adjacent to the removed block will merge with
        # the 0-runs immediately to its left and right.
        # The net effect is:
        #   final_ones = baseline - (ones removed) + (new ones added from zeros)
        #              = baseline - (len(candidate_ones)) + (len(left_zero_run) + len(candidate_ones) + len(right_zero_run))
        #              = baseline + (len(left_zero_run) + len(right_zero_run))
        #
        # So for each eligible ones-run (where runs[i][0] == '1' and i is not the first or last run)
        # we can compute a profit of: left_gap + right_gap.
        #
        # Then the best outcome is: baseline + max(profit) over all eligible candidates.
        max_profit = 0
        # We want to iterate over the runs that are strictly interior.
        for i in range(1, len(runs) - 1):
            # We need a group of ones that are surrounded on both sides by zeros.
            if runs[i][0] == '1':
                # In an alternating run list, the neighbors of a ones-run are zeros-run.
                left_char, left_count = runs[i - 1]
                right_char, right_count = runs[i + 1]
                if left_char == '0' and right_char == '0':
                    profit = left_count + right_count
                    if profit > max_profit:
                        max_profit = profit
        # Only perform the trade if it is beneficial (i.e. if a valid candidate exists).
        return baseline + max_profit if max_profit > 0 else baseline