class Solution:
    def countKConstraintSubstrings(self, s: str, k: int) -> int:
        # We can solve this by considering all substrings and checking if each satisfies the k-constraint.
        # Since |s| <= 50, an O(n^3) solution (by naive checking of each substring) or O(n^2) with prefix sums is feasible.
        
        n = len(s)
        
        # Create prefix sums for 0's and 1's
        prefix_0 = [0] * (n + 1)  # prefix_0[i] = number of '0's in s[:i]
        prefix_1 = [0] * (n + 1)  # prefix_1[i] = number of '1's in s[:i]
        
        for i in range(n):
            prefix_0[i + 1] = prefix_0[i] + (1 if s[i] == '0' else 0)
            prefix_1[i + 1] = prefix_1[i] + (1 if s[i] == '1' else 0)
        
        count_valid = 0
        
        # Check every substring s[i:j]
        for i in range(n):
            for j in range(i+1, n+1):
                # Number of 0's and 1's in s[i:j] can be calculated from prefix sums
                num_0 = prefix_0[j] - prefix_0[i]
                num_1 = prefix_1[j] - prefix_1[i]
                
                # Check if it satisfies having either #0 <= k or #1 <= k
                if num_0 <= k or num_1 <= k:
                    count_valid += 1
        
        return count_valid