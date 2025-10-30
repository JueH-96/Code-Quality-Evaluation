class Solution:
    def countSubstrings(self, s: str, c: str) -> int:
        # Count the occurrences of character c in the string s
        count_c = s.count(c)
        
        # The total number of substrings starting and ending with c
        # can be found using the formula n * (n + 1) // 2
        # where n is the number of occurrences of c.
        return count_c * (count_c + 1) // 2