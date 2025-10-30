class Solution:
    def beautifulIndices(self, s: str, a: str, b: str, k: int) -> List[int]:
        la = len(a)
        lb = len(b)
        ls = len(s)
        
        a_indices = [i for i in range(ls - la + 1) if s[i:i+la] == a]
        b_indices = [i for i in range(ls - lb + 1) if s[i:i+lb] == b]
        
        res = []
        for i in a_indices:
            for j in b_indices:
                if abs(i - j) <= k:
                    res.append(i)
                    break
        res.sort()
        return res