from typing import List

class Solution:
    def validSequence(self, word1: str, word2: str) -> List[int]:
        n = len(word1)
        k = len(word2)

        # Precompute next_occurrence and next_not_occurrence
        # next_occurrence[char_idx][i]: smallest p >= i where word1[p] == char_idx
        # next_not_occurrence[char_idx][i]: smallest p >= i where word1[p] != char_idx
        # Use n + 1 as a sentinel value indicating not found
        next_occurrence = [[n + 1] * (n + 2) for _ in range(26)]
        next_not_occurrence = [[n + 1] * (n + 2) for _ in range(26)]

        for i in range(n - 1, -1, -1):
            char_idx = ord(word1[i]) - ord('a')
            for c in range(26):
                if c == char_idx:
                    next_occurrence[c][i] = i
                    next_not_occurrence[c][i] = next_not_occurrence[c][i+1]
                else:
                    next_occurrence[c][i] = next_occurrence[c][i+1]
                    next_not_occurrence[c][i] = i # Corrected logic: if word1[i] != c, the next != c is at i

        # Ensure next_not_occurrence is correctly built
        for i in range(n - 1, -1, -1):
             char_idx = ord(word1[i]) - ord('a')
             for c in range(26):
                 if c != char_idx:
                      next_not_occurrence[c][i] = i
                 else:
                      next_not_occurrence[c][i] = next_not_occurrence[c][i+1]


        # DP: dp[i][j] = min mismatches to match word2[j..K-1] using K-j chars from word1 at indices >= i
        # Use 2 rows for space optimization: O(K) space.
        # dp[0] for current i, dp[1] for i+1
        # Initialize with 2 (infinity for mismatches)
        dp = [[2] * (k + 1) for _ in range(2)]

        # min_start_idx_le[j][mismatches_budget] = smallest i such that dp[i][j] <= mismatches_budget
        # mismatches_budget can be 0 or 1.
        # Initialize with n + 1 (sentinel value)
        min_start_idx_le = [[n + 1] * 2 for _ in range(k + 1)] # [j][budget]

        # Base case: dp[i][K] = 0 for all 0 <= i <= N.
        # Smallest i >= 0 such that dp[i][K] <= 0 is 0. Smallest i >= 0 such that dp[i][K] <= 1 is 0.
        for i in range(n + 1):
             dp[i % 2][k] = 0

        min_start_idx_le[k][0] = 0
        min_start_idx_le[k][1] = 0


        # Compute DP from j = K-1 down to 0
        for j in range(k - 1, -1, -1):
            curr_row = j % 2
            next_row = (j + 1) % 2

            # Initialize min_start_idx_le for column j
            min_start_idx_le[j][0] = n + 1
            min_start_idx_le[j][1] = n + 1

            # Compute dp[i][j] for i from N down to 0
            # Required length from word1[i:] is K-j. Need N-i >= K-j => i <= N - (K-j)
            # We compute for all i from N down to 0.
            for i in range(n, -1, -1):
                # If not enough characters left in word1[i:]
                if n - i < k - j:
                    dp[curr_row][j] = 2
                else:
                    # Option 1: Skip word1[i]. Need K-j chars from word1[i+1:]
                    # Check if enough chars in word1[i+1:] for K-j
                    if n - (i + 1) >= k - j:
                         skip_cost = dp[next_row][j]
                    else:
                         skip_cost = 2 # effectively infinity

                    # Option 2: Use word1[i] for word2[j]. Need K-j-1 chars from word1[i+1:] for word2[j+1:]
                    # Check if enough chars in word1[i+1:] for K-j-1
                    if n - (i + 1) >= k - (j + 1):
                         use_cost = (word1[i] != word2[j]) + dp[next_row][j + 1]
                         if use_cost > 2: use_cost = 2 # Cap at 2
                    else:
                         use_cost = 2 # effectively infinity

                    dp[curr_row][j] = min(skip_cost, use_cost)

                # Update min_start_idx_le for column j
                # min_start_idx_le[j][b] should be the smallest i such that dp[i][j] <= b
                # Since we iterate i downwards, if dp[i][j] <= b, this `i` is a candidate.
                # As we go further down (smaller i), we find potentially smaller candidates.
                # The *last* `i` encountered where dp[i][j] <= b will be the smallest `i`.
                if dp[curr_row][j] <= 0:
                    min_start_idx_le[j][0] = i 
                if dp[curr_row][j] <= 1:
                    min_start_idx_le[j][1] = i


        # Greedy Algorithm
        result = []
        curr_w1_idx = -1
        mismatches_used = 0

        for j in range(k):
            budget = 1 - mismatches_used
            # Max index in word1 we can pick for word2[j] is N - (K - j)
            search_range_end = n - (k - j) 

            cand_p_exact = n + 1
            if budget >= 0: # Always true if mismatches_used is 0 or 1
                # Need dp[p+1][j+1] <= budget. Smallest i' = p+1 is min_start_idx_le[j+1][budget]
                min_start_for_suffix = min_start_idx_le[j + 1][budget] 
                
                # Need smallest p > curr_w1_idx such that word1[p] == word2[j] and p+1 >= min_start_for_suffix
                # <=> smallest p >= max(curr_w1_idx + 1, min_start_for_suffix - 1) with word1[p] == word2[j]
                search_start_exact = max(curr_w1_idx + 1, min_start_for_suffix - 1)

                if search_start_exact <= n: # Check if search_start_exact is a valid index to start lookup from (indices 0..n are valid start)
                     p = next_occurrence[ord(word2[j]) - ord('a')][search_start_exact]
                     if p <= search_range_end: # Check if found index p is within allowed range for word2[j]
                         cand_p_exact = p


            cand_p_mismatch = n + 1
            if budget >= 1: # Only possible if mismatches_used is 0
                # Need dp[p+1][j+1] <= budget - 1. Smallest i' = p+1 is min_start_idx_le[j+1][budget - 1]
                min_start_for_suffix = min_start_idx_le[j + 1][budget - 1] # budget-1 is 0 here
                
                # Need smallest p > curr_w1_idx such that word1[p] != word2[j] and p+1 >= min_start_for_suffix
                # <=> smallest p >= max(curr_w1_idx + 1, min_start_for_suffix - 1) with word1[p] != word2[j]
                search_start_mismatch = max(curr_w1_idx + 1, min_start_for_suffix - 1)

                if search_start_mismatch <= n: # Check if search_start_mismatch is a valid index to start lookup from
                    p = next_not_occurrence[ord(word2[j]) - ord('a')][search_start_mismatch]
                    if p <= search_range_end: # Check if found index p is within allowed range for word2[j]
                         cand_p_mismatch = p

            next_w1_idx = min(cand_p_exact, cand_p_mismatch)

            if next_w1_idx > n: # No valid index found
                return []

            result.append(next_w1_idx)
            if word1[next_w1_idx] != word2[j]:
                 mismatches_used += 1
            curr_w1_idx = next_w1_idx

        return result