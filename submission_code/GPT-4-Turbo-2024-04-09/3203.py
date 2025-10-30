class Solution:
    def canMakePalindromeQueries(self, s: str, queries: List[List[int]]) -> List[bool]:
        from collections import Counter
        
        # Helper function to count frequencies of characters in a substring
        def count_freq(substring):
            return Counter(substring)
        
        n = len(s)
        half_n = n // 2
        results = []
        
        for a, b, c, d in queries:
            # Count frequencies in the specified ranges
            left_freq = count_freq(s[a:b+1])
            right_freq = count_freq(s[c:d+1])
            
            # Count mismatches
            mismatches = 0
            combined_freq = Counter()
            combined_freq.update(left_freq)
            combined_freq.subtract(right_freq)
            
            # Calculate the number of mismatches (odd counts in combined frequencies)
            for count in combined_freq.values():
                if count % 2 != 0:
                    mismatches += 1
            
            # We can rearrange to form a palindrome if mismatches are zero or can be paired
            results.append(mismatches // 2 <= (b - a + 1) // 2 + (d - c + 1) // 2)
        
        return results