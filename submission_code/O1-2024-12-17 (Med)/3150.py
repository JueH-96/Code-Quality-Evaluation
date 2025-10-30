class Solution:
    def shortestBeautifulSubstring(self, s: str, k: int) -> str:
        n = len(s)
        # Prefix sums to count the number of '1's quickly
        prefix_ones = [0] * (n + 1)
        for i in range(n):
            prefix_ones[i + 1] = prefix_ones[i] + (1 if s[i] == '1' else 0)
        
        # First, find the minimum length of any substring that has exactly k '1's
        min_len = float('inf')
        for start in range(n):
            for end in range(start, n):
                if prefix_ones[end + 1] - prefix_ones[start] == k:
                    length = end - start + 1
                    if length < min_len:
                        min_len = length
        
        # If we never found a substring with exactly k '1's, return ""
        if min_len == float('inf'):
            return ""
        
        # Second, among all substrings of length = min_len with k '1's,
        # find the lexicographically smallest
        best_substring = None
        for start in range(n):
            end = start + min_len - 1
            if end < n:
                if prefix_ones[end + 1] - prefix_ones[start] == k:
                    candidate = s[start:end + 1]
                    if best_substring is None or candidate < best_substring:
                        best_substring = candidate
        
        return best_substring if best_substring else ""