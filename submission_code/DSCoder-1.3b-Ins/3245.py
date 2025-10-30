class Solution:
    def beautifulIndices(self, s: str, a: str, b: str, k: int) -> List[int]:
        n = len(s)
        a_len, b_len = len(a), len(b)
        res = []
        
        for i in range(1, n - a_len + 1):
            if s[i:i+a_len] == a and any(s[j:j+b_len] == b and abs(j - i) <= k for j in range(i+b_len, n - a_len + 1)):
                res.append(i)
        
        return sorted(res)