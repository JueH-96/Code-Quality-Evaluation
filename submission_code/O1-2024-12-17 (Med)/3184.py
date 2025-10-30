class Solution:
    def maxBalancedSubsequenceSum(self, nums: List[int]) -> int:
        # We want a subsequence such that for any two consecutive indices i < j in the subsequence:
        #   nums[j] - nums[i] >= j - i
        #
        # Rearrange:
        #   nums[j] - j >= nums[i] - i
        #
        # Define A[i] = nums[i] - i. Then we need to pick a subsequence where A[i_0] <= A[i_1] <= ... <= A[i_k-1].
        # So we want to find a non-decreasing subsequence in A, maximizing the sum of the corresponding nums values.
        #
        # This can be done using a classic "maximum sum increasing (non-decreasing) subsequence" approach,
        # but instead of "increasing" we allow "A[j] >= A[i]" with i < j. We'll treat it as non-decreasing.
        # We can use a Fenwick (Binary Indexed) Tree or Segment Tree for efficient retrieval of maximum dp
        # among all indices up to A[i]. Then dp[i] = max(0, best_previous_dp) + nums[i].
        # We'll store dp[i] in the Fenwick Tree keyed by the compressed value of A[i].
        
        import math
        
        # Step 1: build array A
        n = len(nums)
        A = [nums[i] - i for i in range(n)]
        
        # Step 2: coordinate compress A
        sorted_unique_A = sorted(set(A))
        rank = {}
        for i, val in enumerate(sorted_unique_A):
            rank[val] = i + 1  # fenwicks are typically 1-indexed
        compressedA = [rank[val] for val in A]
        size = len(sorted_unique_A)
        
        # Fenwick tree (BIT) for maximum values.
        # fenwicks[i] will store the maximum dp value for rank indices up to i.
        fenwicks = [-math.inf] * (size + 1)
        
        def fenwicks_update(i, val):
            while i <= size:
                fenwicks[i] = max(fenwicks[i], val)
                i += i & -i
        
        def fenwicks_query(i):
            res = -math.inf
            while i > 0:
                res = max(res, fenwicks[i])
                i -= i & -i
            return res
        
        # Step 3: DP computation
        # dp[i] = max of dp[j] (where A[j] <= A[i]) + nums[i], or just nums[i] if picking only i.
        # We'll keep track of the global maximum as well.
        
        ans = -math.inf
        
        for i in range(n):
            cidx = compressedA[i]
            best_prev = fenwicks_query(cidx)
            # If best_prev is still -inf, we effectively have no valid previous,
            # so dp[i] can just be nums[i].
            dp_i = max(nums[i], best_prev + nums[i])
            # Update Fenwick
            fenwicks_update(cidx, dp_i)
            # Track global maximum
            ans = max(ans, dp_i)
        
        return ans