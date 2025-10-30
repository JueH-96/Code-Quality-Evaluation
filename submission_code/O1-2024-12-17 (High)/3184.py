class Solution:
    def maxBalancedSubsequenceSum(self, nums: List[int]) -> int:
        """
        We want to find a subsequence i_0 < i_1 < ... < i_{k-1} such that
        nums[i_j] - nums[i_{j-1}] >= (i_j - i_{j-1}) for each j >= 1.
        
        Rearranging nums[i_j] - nums[i_{j-1}] >= i_j - i_{j-1} gives:
            nums[i_j] - i_j >= nums[i_{j-1}] - i_{j-1}.
        
        This means, in the subsequence, the sequence (nums[i] - i) must be
        non-decreasing. Let arr[i] = nums[i] - i.
        
        We then want to choose indices i_0 < i_1 < ... < i_{k-1} so that
        arr[i_0] <= arr[i_1] <= ... <= arr[i_{k-1}],
        and we want to maximize the sum of nums[i_j].
        
        Define dp[i] = maximum possible sum of a balanced subsequence that ends exactly at i.
        Then:
            dp[i] = nums[i] + max(0, max{ dp[j] | j < i and arr[j] <= arr[i] })
        because we can either pick i by itself (if all dp[j] are negative or no j qualifies),
        or extend the best subsequence ending at some j with arr[j] <= arr[i].
        
        To implement dp efficiently (O(n log n)), we use a Fenwick Tree (Binary Indexed Tree)
        over the compressed values of arr, storing and querying the maximum dp value for
        ranks up to a certain point.
        """
        
        import math

        n = len(nums)
        if n == 1:
            return nums[0]  # Only one element => subsequence is that element itself.

        # 1) Build arr[i] = nums[i] - i
        arr = [nums[i] - i for i in range(n)]
        
        # 2) Coordinate compress arr to enable Fenwick Tree indexing
        sorted_unique = sorted(set(arr))
        rank_map = {val: i+1 for i, val in enumerate(sorted_unique)}  # 1-based
        
        # Fenwick tree for maximum queries
        size = len(sorted_unique)
        fenwicks = [float('-inf')] * (size + 1)  # store dp values

        def fenwicks_update(pos, val):
            while pos <= size:
                fenwicks[pos] = max(fenwicks[pos], val)
                pos += pos & -pos

        def fenwicks_query(pos):
            res = float('-inf')
            while pos > 0:
                res = max(res, fenwicks[pos])
                pos -= pos & -pos
            return res
        
        # 3) DP calculation using Fenwick Tree
        dp = [0]*n
        max_sum = float('-inf')
        
        for i in range(n):
            r = rank_map[arr[i]]
            best_prev = fenwicks_query(r)  # max dp among all arr[j] <= arr[i]
            if best_prev < 0:
                best_prev = 0  # Starting a new subsequence if all prev are negative or no valid j
            dp[i] = nums[i] + best_prev
            fenwicks_update(r, dp[i])
            if dp[i] > max_sum:
                max_sum = dp[i]
        
        return max_sum