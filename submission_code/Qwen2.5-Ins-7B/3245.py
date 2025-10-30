class Solution:
    def beautifulIndices(self, s: str, a: str, b: str, k: int) -> List[int]:
        a_indices = []
        b_indices = []
        
        for i in range(len(s) - len(a) + 1):
            if s[i:i+len(a)] == a:
                a_indices.append(i)
        
        for j in range(len(s) - len(b) + 1):
            if s[j:j+len(b)] == b:
                b_indices.append(j)
        
        beautiful_indices = []
        for i in a_indices:
            for j in b_indices:
                if abs(i - j) <= k:
                    beautiful_indices.append(i)
                    break
        
        return sorted(beautiful_indices)