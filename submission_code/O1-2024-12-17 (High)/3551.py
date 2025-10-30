class Solution:
    def maximumSubarrayXor(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        """
        We will precompute the "XOR-score" for every subarray nums[i..j] in O(n^2) using the recurrence:
        
            dp[i][i] = nums[i]
            dp[i][j] = dp[i][j-1] ^ dp[i+1][j]    (for j > i)
        
        Here dp[i][j] is exactly the final single value after repeatedly XOR-merging
        the subarray nums[i..j] until one number remains.

        Once dp is computed, we also build a 'maxFrom[i][j]' = max of dp[i][k] for k from i..j.
        That allows us to answer each query [l,r] by taking:
        
            max( maxFrom[x][r]  for x in [l..r] )
        
        because for each x in [l..r], maxFrom[x][r] is the maximum XOR-score of
        any sub-subarray starting at x and ending no later than r.
        The overall maximum over x in [l..r] covers all sub-subarrays within [l..r].

        Complexity:
          - Building dp takes O(n^2).
          - Building maxFrom takes another O(n^2).
          - Each query then takes O(r-l+1) = O(n) in the worst case.
          - With up to Q = 10^5 queries and n up to 2000, this is about 2*10^8 in the worst case;
            in a lower-level language this can often pass. In Python it is borderline, but we
            implement as efficiently as possible.
        """

        import sys
        input_data = queries  # the problem statement already provides queries as a list of [l, r]

        n = len(nums)
        q = len(queries)

        # -------------------------
        # Step 1: Build dp array
        #   dp[i][j] = final XOR-score for subarray nums[i..j]
        # -------------------------
        dp = [[0]*n for _ in range(n)]
        for i in range(n):
            dp[i][i] = nums[i]
        
        # We fill dp diagonally in descending order of i, ensuring dp[i+1][j] is ready.
        # length = subarray length - 1
        for length in range(1, n):
            for i in range(n - length - 1, -1, -1):
                j = i + length
                dp[i][j] = dp[i][j-1] ^ dp[i+1][j]

        # -------------------------
        # Step 2: Build maxFrom[i][j] = max of dp[i][k] for k in [i..j]
        #         We can fill each row i from left to right.
        # -------------------------
        maxFrom = [[0]*n for _ in range(n)]
        for i in range(n):
            mval = dp[i][i]
            maxFrom[i][i] = mval
            for j in range(i+1, n):
                # maintain the running maximum in row i up to column j
                if dp[i][j] > mval:
                    mval = dp[i][j]
                maxFrom[i][j] = mval

        # -------------------------
        # Step 3: Answer queries
        # For each [l,r], we want
        #    max_{ x in [l..r] } maxFrom[x][r]
        # -------------------------
        answers = [0]*q
        idx = 0
        for (l, r) in queries:
            best = 0
            # scan x from l..r
            for x in range(l, r+1):
                val = maxFrom[x][r]
                if val > best:
                    best = val
            answers[idx] = best
            idx += 1

        return answers