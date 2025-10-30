class Solution:
    def canMakePalindromeQueries(self, s: str, queries: List[List[int]]) -> List[bool]:
        n = len(s)
        ans = []
        for a, b, c, d in queries:
            temp_s = list(s)
            
            sub1 = sorted(temp_s[a:b+1])
            sub2 = sorted(temp_s[c:d+1])
            
            temp_s[a:b+1] = sub1
            temp_s[c:d+1] = sub2