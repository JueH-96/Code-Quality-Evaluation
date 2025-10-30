from typing import List
import math

class Solution:
    def maxScore(self, points: List[int], m: int) -> int:
        n = len(points)

        def check(S: int) -> bool:
            # Calculate the minimum required visits for each index to achieve score S
            # base_c[j] = min_visits[j]
            base_c = [0] * n
            sum_base_c = 0
            for j in range(n):
                # Minimum visits required for index j to get score >= S
                # If S=0, min_visits is 0 as points[j] >= 1.
                # If S > 0, min_visits is ceil(S / points[j]).
                required_visits = (S + points[j] - 1) // points[j] if S > 0 else 0

                # The path must start at index 0, guaranteeing at least 1 visit to index 0.
                # So, the actual minimum required visit count for index 0 is max(1, required_visits).
                # For other indices j > 0, the minimum required visit count is just required_visits.
                if j == 0:
                    base_c[j] = max(1, required_visits)
                else:
                    base_c[j] = required_visits
                sum_base_c += base_c[j]

            # If the total minimum required visits exceed m, score S is impossible.
            if sum_base_c > m:
                return False

            # Calculate the number of extra visits available.
            extra = m - sum_base_c

            # Check path realizability conditions using prefix and suffix sums of visits.
            # A sequence of visit counts c_0, ..., c_{n-1} with sum c_j = m and c_0 >= 1
            # is realizable by a path starting at 0 and staying within [0, n-1] iff:
            # 1. sum_{j=0}^k c_j >= k+1 for k = 0, ..., n-1
            # 2. sum_{j=k}^{n-1} c_j >= n-k for k = 0, ..., n-1
            # We have c_j = base_c[j] + d_j, where d_j are extra visits. sum d_j = extra.
            # Conditions become:
            # 1. sum_{j=0}^k (base_c[j] + d_j) >= k+1  => sum_{j=0}^k d_j >= k+1 - sum_{j=0}^k base_c[j]
            # 2. sum_{j=k}^{n-1} (base_c[j] + d_j) >= n-k => sum_{j=k}^{n-1} d_j >= n-k - sum_{j=k}^{n-1} base_c[j]

            # Let P_k = max(0, k+1 - sum_{j=0}^k base_c[j]) be the minimum required prefix sum of d_j.
            # Let S_k = max(0, n-k - sum_{j=k}^{n-1} base_c[j]) be the minimum required suffix sum of d_j.
            # We need to find non-negative d_j with sum d_j = extra such that:
            # sum_{j=0}^k d_j >= P_k  for k=0..n-1
            # sum_{j=k}^{n-1} d_j >= S_k for k=0..n-1
            # The second condition is equivalent to sum_{j=0}^{k-1} d_j <= extra - S_k for k=1..n.
            # Or sum_{j=0}^k d_j <= extra - S_{k+1} for k=0..n-1 (with S_n = 0).
            # So we need P_k <= sum_{j=0}^k d_j <= extra - S_{k+1} for k=0..n-1.
            # This implies P_k <= extra - S_{k+1} for k=0..n-1.

            # Precompute prefix sums of base_c
            prefix_sum_base = [0] * (n + 1)
            for k in range(n):
                prefix_sum_base[k + 1] = prefix_sum_base[k] + base_c[k]

            # Precompute suffix sums of base_c
            suffix_sum_base = [0] * (n + 1)
            for k in range(n - 1, -1, -1):
                suffix_sum_base[k] = suffix_sum_base[k + 1] + base_c[k]

            for k in range(n):
                # P_k = max(0, k+1 - sum_{j=0}^k base_c[j])
                min_req_d_prefix_k = max(0, (k + 1) - prefix_sum_base[k + 1])

                # S_{k+1} = max(0, n-(k+1) - sum_{j=k+1}^{n-1} base_c[j])
                # sum_{j=k+1}^{n-1} base_c[j] is suffix_sum_base[k+1]
                min_req_d_suffix_k_plus_1 = max(0, (n - (k + 1)) - suffix_sum_base[k + 1])
                
                # Check P_k + S_{k+1} <= extra
                if min_req_d_prefix_k + min_req_d_suffix_k_plus_1 > extra:
                    return False

            return True

        # Binary search for the maximum possible minimum score S.
        low = 0
        # Max possible score is bounded by max(points).
        # Given points[i] <= 10^6, 10^6 is a safe upper bound for the answer.
        # We search in the range [0, 10^6].
        high = 10**6

        ans = 0
        while low <= high:
            mid = low + (high - low) // 2
            if check(mid):
                # mid is a possible score, store it and try higher
                ans = mid
                low = mid + 1
            else:
                # mid is not possible, try lower
                high = mid - 1

        return ans