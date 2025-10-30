class Solution:
    def countSubstrings(self, s: str, c: str) -> int:
        # Count how many times c appears in s
        k = s.count(c)
        # Any substring that starts and ends with c is determined by choosing
        # a pair of occurrences (i, j) with i <= j. That's k choose 2 plus k:
        # k*(k-1)//2 + k = k*(k+1)//2
        return k * (k + 1) // 2