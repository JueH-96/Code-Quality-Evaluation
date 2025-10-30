from typing import List

class Solution:
    def minChanges(self, nums: List[int], k: int) -> int:
        n = len(nums)
        m_pairs = n // 2
        # zero_count[d] = number of pairs with abs(u-v) == d
        zero_count = [0] * (k + 1)
        # freq_m[M] = number of pairs whose one-change max-range M equals this
        freq_m = [0] * (k + 1)

        # Collect zero-change counts and M-values for each pair
        for i in range(m_pairs):
            u = nums[i]
            v = nums[n - 1 - i]
            diff = abs(u - v)
            zero_count[diff] += 1
            # M = max(u, v, k-u, k-v)
            # For X <= M one change suffices; for X > M two changes needed
            M = u
            if v > M: M = v
            ku = k - u
            if ku > M: M = ku
            kv = k - v
            if kv > M: M = kv
            freq_m[M] += 1

        # Build suffix sums of freq_m so that
        # freq_m[x] = number of pairs with M_i >= x (one_possible)
        for x in range(k - 1, -1, -1):
            freq_m[x] += freq_m[x + 1]

        # Compute minimal total cost over X = 0..k
        # cost[X] = 2*m_pairs - one_possible[X] - zero_count[X]
        ans = n  # upper bound: change all elements
        for X in range(k + 1):
            one_possible = freq_m[X]
            zero = zero_count[X]
            cost = 2 * m_pairs - one_possible - zero
            if cost < ans:
                ans = cost

        return ans