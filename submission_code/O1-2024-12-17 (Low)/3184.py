class Solution:
    def maxBalancedSubsequenceSum(self, nums: List[int]) -> int:
        """
        We want a subsequence i0 < i1 < ... < i(k-1) such that for each adjacent pair:
           nums[i_j] - nums[i_{j-1}] >= i_j - i_{j-1}.
        
        Equivalently:
           nums[i_j] - i_j >= nums[i_{j-1}] - i_{j-1}.
        
        We define:
           a[i] = nums[i] - i.
        
        For a subsequence to be balanced when we move from index j to i, we must have:
           a[j] <= a[i].

        We also want to maximize the sum of the elements of such a subsequence.
        Let dp[i] = maximum possible sum of a balanced subsequence that ends exactly at index i.

        Then:
           dp[i] = nums[i] + max(0, max{ dp[j] : a[j] <= a[i], j < i } )
        (The 'max(0, ...)' accounts for the possibility of starting a new subsequence at i.)

        We'll compute dp[i] in an order from left to right and maintain a data structure
        (Fenwick tree or segment tree) that can quickly give us:
           max{ dp[j] : a[j] <= a[i] }
        by storing dp[j] keyed by a[j].

        Steps:
          1) Compute a[i] = nums[i] - i.
          2) Coordinate-compress all a[i] values so they lie in a smaller range [1..m].
          3) Use a Fenwick tree (Binary Indexed Tree) over this compressed range to
             query and update the maximum dp values.
          4) The final answer is max(dp[i]) for i in [0..n-1].
        """

        n = len(nums)
        if n == 1:
            return nums[0]
        
        # Step 1: Build array a = nums[i] - i.
        a = [nums[i] - i for i in range(n)]
        
        # Step 2: Coordinate compression of a[].
        unique_vals = sorted(set(a))
        rank_map = {}
        for idx, val in enumerate(unique_vals):
            rank_map[val] = idx + 1  # ranks start from 1
        
        # Fenwick tree (BIT) for max queries.
        # We'll store at fenwicks[x] = maximum dp among indices with rank <= x.
        # size = len(unique_vals)
        size = len(unique_vals)
        fenwicks = [-10**20] * (size + 1)  # init with very negative
        
        # Define Fenwick update (point update with maximum).
        def update(idx, value):
            while idx <= size:
                fenwicks[idx] = max(fenwicks[idx], value)
                idx += idx & -idx
        
        # Define Fenwick query (max in range [1..idx]).
        def query(idx):
            ans = -10**20
            while idx > 0:
                ans = max(ans, fenwicks[idx])
                idx -= idx & -idx
            return ans
        
        max_sum = -10**20
        
        for i in range(n):
            r = rank_map[a[i]]
            best_prev = query(r)  # max dp[j] over all j with a[j] rank <= a[i]'s rank
            if best_prev < 0:  # if best_prev is negative, we do better by starting new
                best_prev = 0
            dp_i = nums[i] + best_prev
            # Update fenwicks with dp_i at rank r
            update(r, dp_i)
            max_sum = max(max_sum, dp_i)
        
        return max_sum