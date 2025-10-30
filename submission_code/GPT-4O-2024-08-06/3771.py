class Solution:
    def maxSubstringLength(self, s: str, k: int) -> bool:
        # Count the frequency of each character in the string
        from collections import Counter
        char_count = Counter(s)
        
        # Count how many characters appear exactly once
        unique_chars = sum(1 for count in char_count.values() if count == 1)
        
        # We can form at most unique_chars // 2 disjoint special substrings
        # because each special substring must have at least 2 unique characters
        max_special_substrings = unique_chars // 2
        
        # Check if we can form at least k special substrings
        return max_special_substrings >= k