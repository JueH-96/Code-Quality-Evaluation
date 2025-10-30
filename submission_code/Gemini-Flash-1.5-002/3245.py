class Solution:
    def beautifulIndices(self, s: str, a: str, b: str, k: int) -> List[int]:
        indices_a = []
        indices_b = []
        
        for i in range(len(s) - len(a) + 1):
            if s[i:i+len(a)] == a:
                indices_a.append(i)
        
        for i in range(len(s) - len(b) + 1):
            if s[i:i+len(b)] == b:
                indices_b.append(i)
        
        beautiful_indices = []
        for i in indices_a:
            for j in indices_b:
                if abs(i - j) <= k:
                    beautiful_indices.append(i)
                    break
        
        beautiful_indices = sorted(list(set(beautiful_indices)))
        return beautiful_indices