class Solution:
    def minZeroArray(self, nums: List[int], queries: List[List[int]]) -> int:
        n = len(nums)
        m = len(queries)
        required = nums.copy()
        current = [0] * n
        
        for k in range(m):
            l, r, val = queries[k]
            # The query affects indices l to r. For each of these, current[i] can be increased by up to val (but total current[i] must not exceed required[i])
            # However, the actual problem allows selecting any subset, so the optimal is to apply as much as possible per query.
            # So, for each query, the maximum possible is applied to each index in [l, r], but not exceeding remaining (required[i] - current[i])
            # So, the current[i] after processing the query is min(current[i] + val, required[i]) for each i in [l, r]? No, because the query can choose to apply val to any subset.
            # But the problem is to find the minimal k such that after k queries, for each i, the sum of vals from queries (that include i in their [l, r] and were processed up to k) is >= required[i].
            # Wait, no. The actual operation is that for each query, you can select any subset of indices in [l, r], and decrement each by val_i. So the sum of val_i's from queries where i is selected must equal nums[i].
            # So, the minimal k is the earliest point where for each index i, the sum of val's of queries 0..k-1 that include i in their [l, r] is at least nums[i], and the sum is exactly nums[i] for all i.
            pass
        
        # Alternative approach: For each index i, the sum of vals of queries that include i and are in the first k queries must be exactly nums[i], and for all i, this must hold for some k.
        # So, for each index i, collect all queries j that cover i (i.e., l_j <= i <= r_j), ordered by j. Then, the sum of the first s_i queries' vals must equal nums[i].
        # The minimal k is the maximal s_i among all indices, where s_i is the minimal number of queries (in their original order) covering i such that their vals sum to nums[i].
        
        # So, for each index i, we need to process queries in order until the sum of vals of queries that include i reaches nums[i].
        # The overall k is the maximum of these s_i's across all indices, provided that for every i, the sum of the first s_i queries (that include i) equals nums[i].
        # If for any i, the sum of all queries that include i is less than nums[i], return -1.
        
        # So, steps:
        # 1. For each index i, collect all queries (in order) that include i, and compute the prefix sums of their vals.
        # 2. For each i, find the smallest t_i such that the sum of the first t_i vals >= nums[i]. If the sum of all is < nums[i], return -1.
        # 3. The answer is the maximum t_i among all indices.
        
        # But wait: the queries are processed in order, and the subset selection is per query. So for a query j, it can choose to apply to any subset of [l_j, r_j]. So the actual sum for index i is the sum of vals of queries j <= k that include i and were chosen in the subset for j.
        # But the problem is to find the minimal k such that for each i, the sum of vals from queries (up to k) that include i in their subset selections equals nums[i].
        # The minimal k is the earliest query where, for all i, the sum of vals from queries (up to k) that could have been selected for i is at least nums[i], and the sum is exactly nums[i] for all i.
        
        # So, the approach is to model that for each index i, the queries that affect it are those j with l_j <=i <=r_j. The sum of vals of a subset of these queries (selected in order) must equal nums[i]. The minimal k is the latest such query j among all i's minimal required queries.
        
        # So for each index i, we can process the queries in order, and for each query j that includes i, we can add its val to a running sum for i, until the sum reaches nums[i]. The number of queries required for i is the count of such queries processed until the sum is met. The overall answer is the maximum of these counts across all indices.
        # But we must ensure that for each i, the sum of vals of queries that include i is exactly nums[i]. If the sum exceeds for any i, it's impossible.
        
        # Initialize for each index i: remaining = nums[i], and track the queries.
        remaining = nums.copy()
        # For each query in order, for each i in [l, r], subtract val from remaining[i], but only up to remaining[i] >=0.
        # But since in each query, we can choose any subset, the optimal is to subtract as much as possible (i.e., min(val, remaining[i]) for each i in the range.
        # So for each query, the maximum possible is applied to each i in [l, r], which is min(val, remaining[i]).
        # We need to find the earliest k after which all remaining[i] are zero.
        
        for k in range(len(queries)):
            l, r, val = queries[k]
            changed = False
            for i in range(l, r + 1):
                if remaining[i] > 0:
                    delta = min(val, remaining[i])
                    remaining[i] -= delta
                    changed = True
            # After processing this query, check if all remaining are zero
            if all(x == 0 for x in remaining):
                return k + 1
        return -1