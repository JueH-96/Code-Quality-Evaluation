from typing import List

class Solution:
    def maxRemovals(self, source: str, pattern: str, targetIndices: List[int]) -> int:
        n = len(source)
        m = len(pattern)
        # allowed[i] is True if we are allowed to remove source[i] (i.e. i is in targetIndices)
        allowed = [False] * n
        for idx in targetIndices:
            allowed[idx] = True

        # We want to remove as many indices from targetIndices as possible while still 
        # keeping pattern as a subsequence of source.
        #
        # Equivalently, if we plan to remove some indices R ⊆ targetIndices,
        # then the characters kept (i.e. indices not in R) must contain a subsequence that spells pattern.
        # In any valid matching, the indices we use to form the subsequence come from all positions in source.
        # However, if an index is in targetIndices, and we end up having to “protect” it to keep the match,
        # then we cannot remove that index.
        #
        # So if we define each index i with a cost: cost = 1 if i ∈ targetIndices (i.e. if allowed[i] is True)
        # and 0 otherwise, then in forming a subsequence for pattern we would like to pick positions
        # so that the sum of costs (i.e. the number of protected indices from targetIndices) is minimized.
        #
        # Let dp[j] be the minimum “cost” (number of used indices that lie in targetIndices)
        # needed to match pattern[0:j] (j characters) using some increasing index sequence in source.
        # Initially, dp[0] = 0 (we need zero cost to match an empty pattern)
        # and for j>=1, dp[j] = INF.
        #
        # Then, once we compute dp[m] (the minimum cost to match the full pattern),
        # we can remove every allowed index except the ones used in this matching.
        #
        # Hence, the maximum number of removal operations is:
        #     |targetIndices| - (minimum number of indices from targetIndices we had to “protect”).
        
        # For efficient updates we precompute, for each character that appears in pattern, 
        # the list of pattern positions where it occurs.
        char_positions = {}
        for j, ch in enumerate(pattern):
            if ch not in char_positions:
                char_positions[ch] = []
            char_positions[ch].append(j)
        
        INF = 10**9
        dp = [0] + [INF] * m  # dp[0]=0, dp[1..m]=INF

        # Process the source string.
        # For each character in source (at index i), if it equals some pattern letter, we try to use it.
        # When we “use” a character at source index i, we incur a cost:
        #   cost = 1 if i is in targetIndices (i.e. allowed[i] is True), 
        #          0 otherwise.
        # We update in reverse order for each letter so that we do not use the same source index twice.
        for i, ch in enumerate(source):
            cost = 1 if allowed[i] else 0
            if ch in char_positions:
                # Iterate over all positions in pattern where this letter occurs.
                # We update in reverse order to ensure the dp[] we use for updating dp[j+1]
                # comes from an earlier state.
                for j in reversed(char_positions[ch]):
                    if dp[j] != INF:
                        cand = dp[j] + cost
                        if cand < dp[j+1]:
                            dp[j+1] = cand

        # dp[m] is the minimum number of "protected" indices (in targetIndices)
        # required to form pattern as a subsequence.
        # Thus, from the given targetIndices, we may remove the remaining ones.
        return len(targetIndices) - dp[m]