from typing import List

class Solution:
    def maxValue(self, nums: List[int], k: int) -> int:
        n = len(nums)
        if 2 * k > n:
            return 0  # According to constraints, 2*k <= n

        # Precompute dp_first: OR values for exactly m elements up to each index
        dp_first = [{} for _ in range(n)]
        dp_prev = {0: {0}}
        for i in range(n):
            current_dp = {}
            num = nums[i]
            for m_prev in dp_prev:
                for or_prev in dp_prev[m_prev]:
                    # Option 1: do not take num
                    if m_prev in current_dp:
                        current_dp[m_prev].add(or_prev)
                    else:
                        current_dp[m_prev] = {or_prev}
                    # Option 2: take num
                    new_m = m_prev + 1
                    if new_m > k:
                        continue
                    new_or = or_prev | num
                    if new_m in current_dp:
                        current_dp[new_m].add(new_or)
                    else:
                        current_dp[new_m] = {new_or}
            dp_prev = current_dp.copy()
            dp_first[i] = current_dp.copy()

        # Precompute dp_second: OR values for exactly m elements starting from each index
        dp_second = [{} for _ in range(n)]
        dp_next = {0: {0}}
        for j in range(n-1, -1, -1):
            current_dp = {}
            num = nums[j]
            for m_prev in dp_next:
                for or_prev in dp_next[m_prev]:
                    # Option 1: do not take num
                    if m_prev in current_dp:
                        current_dp[m_prev].add(or_prev)
                    else:
                        current_dp[m_prev] = {or_prev}
                    # Option 2: take num
                    new_m = m_prev + 1
                    if new_m > k:
                        continue
                    new_or = or_prev | num
                    if new_m in current_dp:
                        current_dp[new_m].add(new_or)
                    else:
                        current_dp[new_m] = {new_or}
            dp_next = current_dp.copy()
            dp_second[j] = current_dp.copy()

        max_xor = 0
        for i in range(n-1):
            # Check if first group has k elements up to i and second group has k elements starting from i+1
            if k in dp_first[i] and k in dp_second[i+1]:
                for or1 in dp_first[i][k]:
                    for or2 in dp_second[i+1][k]:
                        current_xor = or1 ^ or2
                        if current_xor > max_xor:
                            max_xor = current_xor
        return max_xor