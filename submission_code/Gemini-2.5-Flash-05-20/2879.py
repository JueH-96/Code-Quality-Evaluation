import math

class Solution:
    def minimumChanges(self, s: str, k: int) -> int:
        n = len(s)

        # Precompute costs[i][j]: minimum changes to make s[i...j] a semi-palindrome.
        # costs[i][j] stores the cost for the substring s[i:j+1].
        costs = [[0] * n for _ in range(n)]

        # Iterate through all possible substring lengths.
        # Problem constraints (2 <= s.length, 1 <= k <= s.length / 2) imply
        # that each substring in the partition must have a length of at least 2.
        for length in range(2, n + 1):
            # Iterate through all possible starting indices for a substring of current length.
            for i in range(n - length + 1):
                j = i + length - 1 # Calculate the ending index of the substring s[i...j]
                
                min_changes_for_sub = float('inf')
                
                # Iterate through all possible 'd' values for a semi-palindrome.
                # A string of length L is a semi-palindrome if 1 <= d < L and L % d == 0.
                for d in range(1, length):
                    if length % d == 0:
                        current_changes_for_d = 0
                        sub_seq_len = length // d # Length of the subsequence formed by characters with same modulo d
                        
                        # For each residue class 'r' modulo d (0 to d-1)
                        for r in range(d):
                            # The subsequence is s[i+r], s[i+r+d], s[i+r+2*d], ..., s[i+r+(sub_seq_len-1)*d]
                            # This subsequence must form a palindrome.
                            # We count mismatches by comparing characters from start and end of this subsequence.
                            for p in range(sub_seq_len // 2):
                                char1_idx = i + r + p * d
                                char2_idx = i + r + (sub_seq_len - 1 - p) * d
                                if s[char1_idx] != s[char2_idx]:
                                    current_changes_for_d += 1
                        min_changes_for_sub = min(min_changes_for_sub, current_changes_for_d)
                
                costs[i][j] = min_changes_for_sub

        # dp[k_val][end_idx]: minimum changes to partition s[0...end_idx-1] into k_val substrings.
        # end_idx ranges from 0 to n (exclusive, so up to n for s[0...n-1]).
        dp = [[float('inf')] * (n + 1) for _ in range(k + 1)]

        # Base case: 0 changes to partition an empty string (length 0) into 0 substrings.
        dp[0][0] = 0

        # Fill the dp table
        # num_parts: current number of substrings being formed
        for num_parts in range(1, k + 1):
            # current_end_idx: the exclusive end index of the current prefix s[0...current_end_idx-1].
            # Each substring must have a minimum length of 2.
            # So, current_end_idx must be at least 2 * num_parts.
            for current_end_idx in range(2 * num_parts, n + 1):
                # previous_end_idx: the exclusive end index of the previous prefix s[0...previous_end_idx-1].
                # The last substring is s[previous_end_idx ... current_end_idx-1].
                # This last substring must also have a length of at least 2.
                # previous_end_idx must be at least 2 * (num_parts - 1) to accommodate the
                # preceding num_parts-1 substrings, each of min length 2.
                # previous_end_idx must be at most current_end_idx - 2 to ensure the last substring has length >= 2.
                for previous_end_idx in range(2 * (num_parts - 1), current_end_idx - 1):
                    # Check if the previous state is reachable (not infinity)
                    if dp[num_parts - 1][previous_end_idx] != float('inf'):
                        start_of_last_sub = previous_end_idx
                        end_of_last_sub = current_end_idx - 1
                        
                        # Get the precomputed cost for the last substring.
                        cost_for_last_sub = costs[start_of_last_sub][end_of_last_sub]
                        
                        # Update dp[num_parts][current_end_idx] with the minimum cost.
                        dp[num_parts][current_end_idx] = min(
                            dp[num_parts][current_end_idx],
                            dp[num_parts - 1][previous_end_idx] + cost_for_last_sub
                        )

        # The final answer is the minimum changes to partition the entire string s[0...n-1] into k substrings.
        return dp[k][n]