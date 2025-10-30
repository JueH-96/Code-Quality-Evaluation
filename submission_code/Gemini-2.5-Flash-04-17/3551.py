import math
from typing import List

class Solution:
    def maximumSubarrayXor(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        n = len(nums)

        # Precompute dp table: dp[p][i] = A^(p)[i]
        # where A^(p)[i] is the element at index i after applying the XOR difference
        # operation p times starting from the original nums array.
        # A^(0)[i] = nums[i]
        # A^(p)[i] = A^(p-1)[i] ^ A^(p-1)[i+1] for p > 0
        # Mathematically, A^(p)[i] = sum_{m=0 to p} ((p choose m) mod 2) * nums[i+m]

        # The score of subarray nums[i..j] of length k=j-i+1 is the single element
        # remaining after k-1 steps of the score calculation process.
        # This single element is A^(k-1)[i] = A^(j-i)[i].
        # So, dp[j-i][i] represents the score of nums[i..j].
        
        # Use a list of lists to store dp values.
        # dp[p] stores the values A^(p)[i] for i = 0, ..., n - p - 1.
        # The length of dp[p] is n - p.
        
        dp = [[] for _ in range(n)]
        dp[0] = list(nums)

        # Calculate dp[p] based on dp[p-1]
        for p in range(1, n):
            # A^(p) has length n - p
            dp[p] = [0] * (n - p)
            # Calculate A^(p)[i] for i from 0 to n - p - 1
            # Requires values from A^(p-1) at indices i and i+1.
            # A^(p-1) has length n - (p-1) = n - p + 1.
            # Indices i and i+1 must be valid for A^(p-1), so i+1 <= (n - p + 1) - 1 = n - p.
            # This means i <= n - p - 1.
            for i in range(n - p):
                dp[p][i] = dp[p-1][i] ^ dp[p-1][i+1]

        # Precompute log_2 table for Sparse Table
        # log_table[k] stores floor(log2(k))
        log_table = [0] * (n + 1)
        # Handle k=1 case separately if needed, but loop starts from 2
        # log_table[1] remains 0
        for i in range(2, n + 1):
            log_table[i] = log_table[i // 2] + 1

        # Precompute Sparse Tables for each row of dp table
        # ST[p] is the sparse table for dp[p]
        # ST[p][j][i] stores max value in dp[p][i .. i + 2^j - 1]
        # This allows O(1) range maximum queries on each row.
        
        ST = [[] for _ in range(n)]

        for p in range(n):
            row_len = n - p
            if row_len <= 0: # Should only happen if n=0, but constraints say n >= 1
                continue
            
            # Max power of 2 <= row_len. j goes from 0 up to max_j.
            # If row_len is 1, log_table[1] is 0, max_j is 0. Range j=0..0.
            max_j = log_table[row_len]
            
            # ST[p] has dimensions (max_j + 1) rows and row_len columns
            ST[p] = [[0] * row_len for _ in range(max_j + 1)]

            # Base case j = 0: range length 2^0 = 1. ST[p][0][i] is just dp[p][i].
            for i in range(row_len):
                ST[p][0][i] = dp[p][i]
            
            # Fill sparse table for j > 0 using values from j-1
            for j in range(1, max_j + 1):
                # A range of length 2^j starting at i ends at i + 2^j - 1.
                # This end index must be within the row bounds [0, row_len - 1].
                # So, i + 2^j - 1 < row_len => i < row_len - 2^j + 1.
                for i in range(row_len - (1 << j) + 1):
                    # Max of two segments of length 2^(j-1) covering the current segment
                    ST[p][j][i] = max(ST[p][j-1][i], ST[p][j-1][i + (1 << (j-1))])

        # Function to query max in range [L, R] (inclusive) in dp[p] using ST[p]
        def query_st(p, L, R):
            # Ensure the range is valid for dp[p]
            # The query loop logic ensures L <= R and 0 <= L <= n-p-1.
            
            length = R - L + 1
            # Using precomputed log_table for O(1) log base 2
            j = log_table[length]
            
            # The maximum in the range [L, R] is the maximum of two overlapping segments
            # of length 2^j that together cover the range.
            # Segment 1: starts at L, length 2^j. Range [L, L + 2^j - 1]
            # Segment 2: ends at R, length 2^j. Starts at R - 2^j + 1. Range [R - 2^j + 1, R]
            return max(ST[p][j][L], ST[p][j][R - (1 << j) + 1])

        # Process queries
        results = []
        for l, r in queries:
            max_score = 0
            
            # For a query [l, r], we consider all subarrays nums[i..j] where l <= i <= j <= r.
            # The score of nums[i..j] is dp[j-i][i]. Let p = j - i.
            # We need to find the maximum of dp[p][i] for all valid pairs (p, i).
            # Constraints on (i, j) are l <= i <= j <= r.
            # Constraints on (p, i) derived from (i, j):
            # 1. p = j - i >= 0 (since j >= i)
            # 2. From l <= i, we have i >= l.
            # 3. From j <= r, we have i + p <= r, so i <= r - p.
            # 4. The maximum possible value of p is when i = l and j = r, so p_max = r - l.
            # Combining constraints on p and i:
            # 0 <= p <= r - l
            # l <= i <= r - p

            # Iterate over possible values of p (the difference j - i)
            # p ranges from 0 (subarray length 1) to r - l (subarray length r - l + 1)
            for p in range(r - l + 1):
                # For a fixed p, the starting index i must satisfy l <= i <= r - p.
                start_idx_range_l = l
                start_idx_range_r = r - p

                # The score values dp[p][i] for fixed p and i in the range [start_idx_range_l, start_idx_range_r]
                # are stored contiguously in the list dp[p] at those indices.
                # We need the maximum value in the range dp[p][start_idx_range_l .. start_idx_range_r].
                current_max = query_st(p, start_idx_range_l, start_idx_range_r)
                max_score = max(max_score, current_max)

            results.append(max_score)

        return results