class Solution:
    def canMakePalindromeQueries(self, s: str, queries: List[List[int]]) -> List[bool]:
        n = len(s)
        half = n // 2
        
        # Precompute prefix counts for each character in both halves
        prefix_left = [[0] * 26 for _ in range(half + 1)]
        prefix_right = [[0] * 26 for _ in range(half + 1)]
        
        for i in range(half):
            for j in range(26):
                prefix_left[i + 1][j] = prefix_left[i][j]
                prefix_right[i + 1][j] = prefix_right[i][j]
            prefix_left[i + 1][ord(s[i]) - ord('a')] += 1
            prefix_right[i + 1][ord(s[n - 1 - i]) - ord('a')] += 1
        
        # Function to check if two ranges have the same character counts
        def same_counts(l1, r1, l2, r2):
            return all(prefix_left[r1 + 1][i] - prefix_left[l1][i] == 
                       prefix_right[r2 + 1][i] - prefix_right[l2][i] 
                       for i in range(26))
        
        results = []
        for a, b, c, d in queries:
            c, d = n - 1 - d, n - 1 - c  # Convert to left half indices
            
            # Check if parts outside query ranges are palindromes
            if not (same_counts(0, min(a, c) - 1, 0, min(a, c) - 1) and
                    same_counts(max(b, d) + 1, half - 1, max(b, d) + 1, half - 1)):
                results.append(False)
                continue
            
            # Check if query ranges cover all mismatched parts
            if a <= c:
                if b >= d:
                    results.append(same_counts(a, b, c, d))
                else:
                    results.append(same_counts(a, d, a, d) and same_counts(b + 1, c - 1, b + 1, c - 1))
            else:
                if d >= b:
                    results.append(same_counts(c, d, a, b))
                else:
                    results.append(False)
        
        return results