import bisect

class Solution:
    def beautifulIndices(self, s: str, a: str, b: str, k: int) -> List[int]:
        n = len(s)
        len_a = len(a)
        len_b = len(b)
        
        indices_a = []
        for i in range(0, n - len_a + 1):
            if s[i:i+len_a] == a:
                indices_a.append(i)
                
        indices_b = []
        for j in range(0, n - len_b + 1):
            if s[j:j+len_b] == b:
                indices_b.append(j)
                
        if not indices_b:
            return []
        
        res = []
        for i in indices_a:
            low_bound = i - k
            high_bound = i + k
            pos = bisect.bisect_left(indices_b, low_bound)
            if pos < len(indices_b) and indices_b[pos] <= high_bound:
                res.append(i)
                
        return res