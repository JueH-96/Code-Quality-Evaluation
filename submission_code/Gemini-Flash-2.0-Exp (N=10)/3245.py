class Solution:
    def beautifulIndices(self, s: str, a: str, b: str, k: int) -> List[int]:
        n = len(s)
        len_a = len(a)
        len_b = len(b)
        
        a_indices = []
        for i in range(n - len_a + 1):
            if s[i:i+len_a] == a:
                a_indices.append(i)
        
        b_indices = []
        for i in range(n - len_b + 1):
            if s[i:i+len_b] == b:
                b_indices.append(i)
        
        beautiful_indices = []
        for i in a_indices:
            for j in b_indices:
                if abs(j - i) <= k:
                    beautiful_indices.append(i)
                    break
        
        beautiful_indices.sort()
        return beautiful_indices