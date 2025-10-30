import math
from typing import List

class Solution:
    def maxScore(self, points: List[int], m: int) -> int:
        n = len(points)

        # Binary search for the maximum possible minimum score X.
        # The minimum score can be 0.
        # A safe upper bound for the maximum possible minimum score X
        # is the maximum value in points. If X > max(points), then for
        # the minimum point value, ceil(X / points[i]) >= 2 (unless points[i] is huge),
        # increasing required visits quickly.
        # A loose but sufficient upper bound based on constraints: max(points) + m/n (rough visits per index)
        # but m is large. max(points) should be enough as minimum score.
        # If X > max(points), then for any index i with points[i] = min(points),
        # the required visits c[i] = ceil(X / points[i]) will be > 1.
        # If X is the minimum score, X must be achievable.
        # Max possible score for a single index visited roughly m/2 times is m/2 * max(points).
        # But we want the minimum among all scores.
        # The maximum possible minimum score cannot exceed max(points). If the minimum score was S > max(points),
        # then for the index i with points[i] = min(points), we would need ceil(S/points[i]) visits.
        # If S > max(points) >= points[i], ceil(S/points[i]) >= 2.
        # Let's take a conservative upper bound: max(points) + 1 is sufficient because if X > max(points), then for any i, ceil(X/points[i]) * points[i] > points[i].
        # But even better: max(points) + m could be an upper bound if m is small.
        # If X = max(points) + 1, minimum visits for points[i]=1 is X.
        # A safe upper bound on X is max(points) + 1 (if m >= n), or more generally, the highest score reachable at a single index, which is bounded by max(points) * m.
        # However, since we need the MINIMUM score to be X, X is likely bounded by max(points).
        # If minimum score is X, then for any index i, points[i] * visits[i] >= X.
        # If X > max(points), then for all i, points[i] < X.
        # The number of visits c[i] = ceil(X/points[i]) is always >= 1.
        # Total visits >= n. Min moves >= n.
        # Binary search range [0, max(points) + 1] should be sufficient for many cases.
        # Let's use a larger safe upper bound like 2 * 10^6 just to be safe.

        low = 0
        high = int(1e6 + 7) # A sufficiently large upper bound for the minimum score

        ans = 0

        def check(X: int) -> bool:
            # Can we achieve a minimum score of at least X using at most m moves?
            if X < 0:
                return True # Trivial case, minimum score 0 is always possible (0 moves leaves scores at 0).
            
            # required visits for each index i to achieve score >= X
            c = [0] * n
            for i in range(n):
                # ceil(X / points[i])
                # If X is 0, c[i] should be 0. (0 + points[i] - 1) // points[i] is 0 if X is 0.
                c[i] = (X + points[i] - 1) // points[i]
            
            # Problem starts at index -1. First move must be -1 -> 0.
            # This costs 1 move and visits index 0 once.
            # We have m-1 moves left, starting at index 0.
            # We need total visits v_j >= c[j] for all j.
            # Since we visit index 0 once initially, we need additional visits:
            # c'_0 = max(0, c_0 - 1)
            # c'_j = c_j for j > 0
            
            c_prime = [0] * n
            c_prime[0] = max(0, c[0] - 1)
            for i in range(1, n):
                c_prime[i] = c[i]

            # Minimum path length k' required starting at index 0 using m-1 moves
            # to achieve visit counts v_j >= c'_j for all j.
            # This is a known problem on path graphs.
            # The minimum path length k' from vertex 0 to achieve visit counts >= c'[j] is:
            # k' >= max_{0 <= i < n} (sum_{j=0..i} c'[j] + i) AND
            # k' >= max_{0 <= i < n} (sum_{j=i..n-1} c'[j] + (n-1-i)) # This is min steps to end at i ? No.

            # Let k' be the minimum steps needed starting from index 0.
            # k' must be at least sum(c').
            # For any prefix [0, i], the path must spend enough time there to achieve required visits.
            # Min steps k' >= max_{0 <= i < n} (sum_{j=0..i} c'[j] + i) seems to be wrong.

            # Let P'[i] = sum_{j=0..i} c'[j]
            # Let S'[i] = sum_{j=i..n-1} c'[j]
            # Minimum path length k' from 0 to get counts >= c'[j] is max(max_{0<=i<n} (P'[i] + i), max_{0<=i<n} (S'[i] + (n-1-i))). This is wrong.

            # The correct formula for minimum path length k from vertex 0 on path graph 0..n-1
            # to achieve visit counts c[j] >= required_c[j]:
            # k = max( max_{0 <= i < n} ( sum_{j=0..i} required_c[j] + i), max_{0 <= i < n} ( sum_{j=i..n-1} required_c[j] + (n-1-i) ) )
            # This formula calculates the minimum steps for a path that visits all required nodes in [0, i] and ends at i, or visits all required nodes in [i, n-1] and ends at i.

            # Let's calculate the two bounds based on c_prime
            
            # Bound 1: Maximum of (cumulative required visits from left + distance from 0)
            min_k_prime_bound1 = 0
            current_prefix_sum_c_prime = 0
            for i in range(n):
                 current_prefix_sum_c_prime += c_prime[i]
                 min_k_prime_bound1 = max(min_k_prime_bound1, current_prefix_sum_c_prime + i)

            # Bound 2: Maximum of (cumulative required visits from right + distance to n-1) + distance from 0 to i?
            # Min steps to cover [i, n-1] with counts c'[i]..c'[n-1] starting at 0
            # Path must reach i (i steps), then cover [i, n-1] ending at n-1
            # Min steps to cover [i, n-1] ending at n-1 is sum_{j=i..n-1} c'[j] + (n-1-i)
            # Total min steps from 0 to cover [i, n-1] ending at n-1 is i + sum_{j=i..n-1} c'[j] + (n-1-i) ? No.

            # The other term in the max is max_{0 <= i < n} (sum_{j=i..n-1} c'[j] + (n-1-i)).
            # This represents the minimum steps to visit the required counts in [i, n-1] ending at index i, starting at index i.
            # Since we start at 0, we need i additional steps to reach i.

            min_k_prime_bound2 = 0
            current_suffix_sum_c_prime = 0
            for i in range(n - 1, -1, -1):
                current_suffix_sum_c_prime += c_prime[i]
                min_k_prime_bound2 = max(min_k_prime_bound2, current_suffix_sum_c_prime + (n - 1 - i) + i) # This is wrong.

            # Correct bounds for min length k from vertex 0 to get counts >= c[j]:
            # k >= max_{0 <= i < n} (sum_{j=0..i} c[j] + i) AND
            # k >= max_{0 <= i < n} (sum_{j=i..n-1} c[j] + (n-1-i))
            # The overall minimum k is the max of these two values.

            min_k_prime_val1 = 0
            curr_p_sum = 0
            for i in range(n):
                 curr_p_sum += c_prime[i]
                 min_k_prime_val1 = max(min_k_prime_val1, curr_p_sum + i)

            min_k_prime_val2 = 0
            curr_s_sum = 0
            for i in range(n - 1, -1, -1):
                curr_s_sum += c_prime[i]
                min_k_prime_val2 = max(min_k_prime_val2, curr_s_sum + (n - 1 - i)) # This is dist to n-1 if ending at i

            min_k_prime = max(min_k_prime_val1, min_k_prime_val2)

            return min_k_prime <= m - 1


        while low <= high:
            mid = (low + high) // 2
            if check(mid):
                ans = mid
                low = mid + 1
            else:
                high = mid - 1

        return ans