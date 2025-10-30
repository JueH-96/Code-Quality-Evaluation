class Solution:
    def minAnagramLength(self, s: str) -> int:
        # To find the minimum possible length of the string t,
        # we need to find the number of unique characters in s.
        # This is because the minimum t would be composed of all unique characters
        # from s, and any anagram of t would also be a valid substring of s.
        
        # Use a set to find the unique characters in s
        unique_chars = set(s)
        
        # The length of the set gives the minimum possible length of t
        return len(unique_chars)