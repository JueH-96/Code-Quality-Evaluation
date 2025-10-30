class Solution:
    def beautifulIndices(self, s: str, a: str, b: str, k: int) -> List[int]:
        def find_indices(s, pattern):
            indices = []
            start = 0
            while start < len(s):
                start = s.find(pattern, start)
                if start == -1:
                    break
                indices.append(start)
                start += 1
            return indices
        
        a_indices = find_indices(s, a)
        b_indices = find_indices(s, b)
        
        beautiful_indices = []
        for i in a_indices:
            for j in b_indices:
                if abs(j - i) <= k:
                    beautiful_indices.append(i)
                    break
        
        return beautiful_indices