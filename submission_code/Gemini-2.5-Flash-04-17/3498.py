from typing import List
import sys

class Solution:
    def minChanges(self, nums: List[int], k: int) -> int:
        n = len(nums)
        # count_diff[d] stores the number of pairs (nums[i], nums[n-i-1]) with absolute difference d.
        # The possible differences are from 0 to k.
        count_diff = [0] * (k + 1)
        
        # count_mr[v] stores the number of pairs (nums[i], nums[n-i-1])
        # where max_reach_one_change is equal to v.
        # max_reach_one_change for a pair (a, b) is max(max(a, k-a), max(b, k-b)).
        # The minimum value of max(x, k-x) for x in [0, k] is k/2 (when x=k/2).
        # The maximum value is k (when x=0 or x=k).
        # So max_reach_one_change is in [k/2, k]. Array size k+1 is sufficient.
        count_mr = [0] * (k + 1)

        for i in range(n // 2):
            a = nums[i]
            b = nums[n - 1 - i]

            # Original difference for this pair.
            diff = abs(a - b)
            count_diff[diff] += 1

            # max_reach_one_change is the maximum possible absolute difference X
            # such that we can achieve difference X by changing *exactly one* element
            # in the pair (a, b) to a value in [0, k].
            max_reach_one_change = max(max(a, k - a), max(b, k - b))
            count_mr[max_reach_one_change] += 1

        # Calculate count_lt_mr[X] = number of pairs with max_reach_one_change < X.
        # This is the prefix sum of count_mr.
        # count_lt_mr[X] = sum_{v=0}^{X-1} count_mr[v].
        # We need this value for X from 0 to k.
        count_lt_mr = [0] * (k + 1)
        
        # count_lt_mr[0] is sum up to -1, which is 0.
        # count_lt_mr[X] for X > 0 is count_lt_mr[X-1] + count_mr[X-1].
        for X in range(1, k + 1):
             count_lt_mr[X] = count_lt_mr[X-1] + count_mr[X-1]

        # Total cost for a target difference X is calculated as follows:
        # For a pair i with original difference diff_i and max_reach_one_change mr_i:
        # - Cost is 0 if X == diff_i
        # - Cost is 1 if X != diff_i and X <= mr_i
        # - Cost is 2 if X != diff_i and X > mr_i
        # (Note: diff_i <= mr_i is always true)
        # Summing over all pairs:
        # TotalCost(X) = Sum_{i: diff_i == X} 0 + Sum_{i: diff_i != X, mr_i >= X} 1 + Sum_{i: diff_i != X, mr_i < X} 2
        # TotalCost(X) = |{i | diff_i != X, mr_i >= X}| * 1 + |{i | diff_i != X, mr_i < X}| * 2
        # TotalCost(X) = |{i | mr_i >= X}| * 1 + |{i | mr_i < X}| * 2 - |{i | diff_i == X and mr_i >= X}| * 1 (Correction for cost 0)
        # Since diff_i <= mr_i, if diff_i == X, then mr_i >= X is always true.
        # TotalCost(X) = |{i | mr_i >= X}| + 2 * |{i | mr_i < X}| - |{i | diff_i == X}|
        # TotalCost(X) = (n/2 - |{i | mr_i < X}|) + 2 * |{i | mr_i < X}| - count_diff[X]
        # TotalCost(X) = n/2 + |{i | mr_i < X}| - count_diff[X]
        # TotalCost(X) = n/2 + count_lt_mr[X] - count_diff[X]

        min_changes = sys.maxsize

        # Iterate through all possible target differences X from 0 to k.
        for X in range(k + 1):
            # count_lt_mr[X] is the number of pairs with mr_i < X.
            # count_diff[X] is the number of pairs with diff_i = X.
            current_changes = (n // 2) + count_lt_mr[X] - count_diff[X]
            min_changes = min(min_changes, current_changes)

        return min_changes