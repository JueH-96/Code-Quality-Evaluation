class Solution:
    def countSubstrings(self, s: str, c: str) -> int:
        # Count the number of occurrences of character c in string s
        count = s.count(c)
        
        # Use the formula: count * (count + 1) // 2
        return count * (count + 1) // 2