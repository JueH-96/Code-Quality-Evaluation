class Solution:
    def countSubstrings(self, s: str, c: str) -> int:
        # Collect all indices where character c appears in s
        indices = [i for i, char in enumerate(s) if char == c]
        
        # The number of substrings that start and end with c is the number
        # of ways we can pick two occurrences of c (start, end), plus the single-character substrings.
        # This is simply combination of k positions taken two at a time plus k (or the formula k*(k+1)//2).
        k = len(indices)
        return k * (k + 1) // 2