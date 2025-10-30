from typing import List
import bisect
from collections import Counter, defaultdict

class Solution:
    def maxSubarrays(self, n: int, conflictingPairs: List[List[int]]) -> int:
        m = len(conflictingPairs)
        A_to_s = defaultdict(list)
        for idx, pair in enumerate(conflictingPairs):
            a, b = pair
            A = min(a, b)
            B = max(a, b)
            A_to_s[A].append(idx)

        intervals = []
        for idx, pair in enumerate(conflictingPairs):
            a, b = pair
            A = min(a, b)
            B = max(a, b)
            intervals.append((B, A, idx))  # B, A, original idx
        intervals.sort()  # sort by B asc

        counter_A = Counter()
        sorted_A_values = []
        increase = [0] * m
        sum_good = 0
        idx_interval = 0

        for R in range(1, n + 1):
            # Add all intervals with B = R
            while idx_interval < m and intervals[idx_interval][0] == R:
                B, A_t, s_idx = intervals[idx_interval]
                if A_t not in counter_A:
                    ins_index = bisect.bisect_left(sorted_A_values, A_t)
                    sorted_A_values.insert(ins_index, A_t)
                    counter_A[A_t] = 1
                else:
                    counter_A[A_t] += 1
                idx_interval += 1

            # After adding, compute M(R) etc.
            if len(sorted_A_values) == 0:
                max1 = 0
                count_max = 0
                second_max = 0
            else:
                max1 = sorted_A_values[-1]
                count_max = counter_A[max1]
                if count_max > 1:
                    second_max = max1
                else:
                    if len(sorted_A_values) > 1:
                        second_max = sorted_A_values[-2]
                    else:
                        second_max = 0

            # Add to sum_good
            sum_good += max(0, R - max1)

            # If count_max == 1:
            if count_max == 1:
                s_idx = A_to_s[max1][0]  # the index of s with A = max1
                A_s = max1
                if R > A_s:
                    difference = A_s - second_max
                    increase[s_idx] += difference

        # Now after all R, good_full = sum_good
        good_full = sum_good
        # Compute max over s of good_full + increase[s]
        ans = 0
        for s in range(m):
            good_s = good_full + increase[s]
            if good_s > ans:
                ans = good_s
        return ans