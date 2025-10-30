class Solution:
    def shortestBeautifulSubstring(self, s: str, k: int) -> str:
        n = len(s)
        # Precompute prefix sums of the count of '1's for O(1) range queries
        # prefix[i] = number of '1's up to (but not including) index i
        prefix = [0] * (n + 1)
        for i in range(n):
            prefix[i+1] = prefix[i] + (1 if s[i] == '1' else 0)
        
        min_length = float('inf')
        result = ""
        
        # Check all substrings
        for start in range(n):
            for end in range(start, n):
                count_ones = prefix[end + 1] - prefix[start]
                if count_ones == k:
                    length = end - start + 1
                    if length < min_length:
                        min_length = length
                        result = s[start:end+1]
                    elif length == min_length:
                        # Update result if we find a lexicographically smaller substring
                        candidate = s[start:end+1]
                        if candidate < result:
                            result = candidate
        
        return result if min_length != float('inf') else ""