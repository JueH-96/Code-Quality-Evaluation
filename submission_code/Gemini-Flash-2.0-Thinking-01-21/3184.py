import collections

class SegmentTree:
    def __init__(self, n):
        self.n = n
        self.tree = [-float('inf')] * (4 * n)

    def update(self, index, value):
        self._update_recursive(0, 0, self.n - 1, index, value)

    def _update_recursive(self, node_index, start, end, target_index, value):
        if start == end:
            self.tree[node_index] = max(self.tree[node_index], value)
            return
        mid = (start + end) // 2
        if start <= target_index <= mid:
            self._update_recursive(2 * node_index + 1, start, mid, target_index, value)
        else:
            self._update_recursive(2 * node_index + 2, mid + 1, end, target_index, value)
        self.tree[node_index] = max(self.tree[2 * node_index + 1], self.tree[2 * node_index + 2])

    def query_max(self, query_end_index):
        if query_end_index < 0:
            return -float('inf')
        return self._query_max_recursive(0, 0, self.n - 1, 0, query_end_index)

    def _query_max_recursive(self, node_index, start, end, query_start, query_end):
        if query_end < start or end < query_start:
            return -float('inf')
        if query_start <= start and end <= query_end:
            return self.tree[node_index]
        mid = (start + end) // 2
        left_max = self._query_max_recursive(2 * node_index + 1, start, mid, query_start, query_end)
        right_max = self._query_max_recursive(2 * node_index + 2, mid + 1, end, query_start, query_end)
        return max(left_max, right_max)

class Solution:
    def maxBalancedSubsequenceSum(self, nums: List[int]) -> int:
        n = len(nums)
        diffs = [nums[i] - i for i in range(n)]
        unique_diffs = sorted(list(set(diffs)))
        diff_rank_map = {diff_val: rank for rank, diff_val in enumerate(unique_diffs)}
        ranks = [diff_rank_map[diff] for diff in diffs]
        num_unique_diffs = len(unique_diffs)
        st = SegmentTree(num_unique_diffs)
        dp = [0] * n
        for i in range(n):
            rank = ranks[i]
            max_prev_dp = st.query_max(rank)
            current_dp = nums[i] + max(0, max_prev_dp)
            dp[i] = current_dp
            st.update(rank, current_dp)
        max_sum = max(dp) if dp else -float('inf')
        if max_sum < 0:
            return max(nums) if max(nums) <= 0 else max_sum
        return max_sum if max_sum > -float('inf') else max(nums) if nums else 0

from typing import List

class Solution:
    def maxBalancedSubsequenceSum(self, nums: List[int]) -> int:
        n = len(nums)
        diffs = [nums[i] - i for i in range(n)]
        unique_diffs = sorted(list(set(diffs)))
        diff_rank_map = {diff_val: rank for rank, diff_val in enumerate(unique_diffs)}
        ranks = [diff_rank_map[diff] for diff in diffs]
        num_unique_diffs = len(unique_diffs)
        dp = [0] * n
        rank_max_dp = collections.defaultdict(lambda: -float('inf'))
        max_overall_sum = -float('inf')
        for i in range(n):
            rank = ranks[i]
            max_prev_dp = -float('inf')
            for r in range(rank + 1):
                max_prev_dp = max(max_prev_dp, rank_max_dp[r])
            current_dp = nums[i] + max(0, max_prev_dp)
            dp[i] = current_dp
            rank_max_dp[rank] = max(rank_max_dp[rank], current_dp)
            max_overall_sum = max(max_overall_sum, current_dp)
            
        if max_overall_sum == -float('inf'):
            return max(nums) if nums else 0
        return max(max_overall_sum, max(x for x in nums if x >= 0) if any(x >= 0 for x in nums) else max(nums) if nums else 0)

class SegmentTree:
    def __init__(self, n):
        self.n = n
        self.tree = [-float('inf')] * (4 * n)

    def update(self, index, value):
        self._update_recursive(0, 0, self.n - 1, index, value)

    def _update_recursive(self, node_index, start, end, target_index, value):
        if start == end:
            self.tree[node_index] = max(self.tree[node_index], value)
            return
        mid = (start + end) // 2
        if start <= target_index <= mid:
            self._update_recursive(2 * node_index + 1, start, mid, target_index, value)
        else:
            self._update_recursive(2 * node_index + 2, mid + 1, end, target_index, value)
        self.tree[node_index] = max(self.tree[2 * node_index + 1], self.tree[2 * node_index + 2])

    def query_max(self, query_end_index):
        if query_end_index < 0:
            return -float('inf')
        return self._query_max_recursive(0, 0, self.n - 1, 0, query_end_index)

    def _query_max_recursive(self, node_index, start, end, query_start, query_end):
        if query_end < start or end < query_start:
            return -float('inf')
        if query_start <= start and end <= query_end:
            return self.tree[node_index]
        mid = (start + end) // 2
        left_max = self._query_max_recursive(2 * node_index + 1, start, mid, query_start, query_end)
        right_max = self._query_max_recursive(2 * node_index + 2, mid + 1, end, query_start, query_end)
        return max(left_max, right_max)

class Solution:
    def maxBalancedSubsequenceSum(self, nums: List[int]) -> int:
        n = len(nums)
        diffs = [nums[i] - i for i in range(n)]
        unique_diffs = sorted(list(set(diffs)))
        diff_rank_map = {diff_val: rank for rank, diff_val in enumerate(unique_diffs)}
        ranks = [diff_rank_map[diff] for diff in diffs]
        num_unique_diffs = len(unique_diffs)
        st = SegmentTree(num_unique_diffs)
        dp = [0] * n
        max_overall_sum = -float('inf')
        for i in range(n):
            rank = ranks[i]
            max_prev_dp = st.query_max(rank)
            current_dp = nums[i] + max(0, max_prev_dp)
            dp[i] = current_dp
            st.update(rank, current_dp)
            max_overall_sum = max(max_overall_sum, current_dp)
            
        if max_overall_sum == -float('inf'):
            return max(nums) if nums else 0
        return max_overall_sum