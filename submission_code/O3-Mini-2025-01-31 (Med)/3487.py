from typing import List

class Solution:
    def maxRemovals(self, source: str, pattern: str, targetIndices: List[int]) -> int:
        n, m = len(source), len(pattern)
        INF = float('inf')
        # Convert targetIndices to a set for O(1) membership tests.
        candidate_set = set(targetIndices)
        # dp[i] is the minimum number of candidate positions that must be kept
        # (i.e. not removed) in order to form the subsequence pattern[0:i]
        dp = [INF] * (m + 1)
        dp[0] = 0  # matching empty pattern requires 0 kept candidates

        # Process the source string from left to right.
        # For each character, if it can help match the next character in the pattern,
        # update the dp state. If the index is in candidate_set, then keeping it "costs" 1;
        # otherwise (index not removable) cost is 0.
        for j in range(n):
            cost = 1 if j in candidate_set else 0
            # Update backwards to prevent using this letter more than once in the same step.
            for i in range(m - 1, -1, -1):
                if source[j] == pattern[i] and dp[i] != INF:
                    dp[i + 1] = min(dp[i + 1], dp[i] + cost)
                    
        # dp[m] is the minimum candidate indices that we must keep in order to have pattern as a subsequence.
        # Since we are only allowed to remove letters at indices in targetIndices, the maximum removals
        # equals the total candidate positions minus the ones that we must keep.
        total_candidates = len(targetIndices)
        return total_candidates - dp[m]


# For testing the solution with the provided examples:
if __name__ == "__main__":
    sol = Solution()
    
    # Example 1:
    source = "abbaa"
    pattern = "aba"
    targetIndices = [0, 1, 2]
    print(sol.maxRemovals(source, pattern, targetIndices))  # Expected output: 1

    # Example 2:
    source = "bcda"
    pattern = "d"
    targetIndices = [0, 3]
    print(sol.maxRemovals(source, pattern, targetIndices))  # Expected output: 2

    # Example 3:
    source = "dda"
    pattern = "dda"
    targetIndices = [0, 1, 2]
    print(sol.maxRemovals(source, pattern, targetIndices))  # Expected output: 0

    # Example 4:
    source = "yeyeykyded"
    pattern = "yeyyd"
    targetIndices = [0, 2, 3, 4]
    print(sol.maxRemovals(source, pattern, targetIndices))  # Expected output: 2