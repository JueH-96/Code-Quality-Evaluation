class Solution:
    def beautifulIndices(self, s: str, a: str, b: str, k: int) -> List[int]:
        len_s = len(s)
        len_a = len(a)
        len_b = len(b)
        
        a_indices = []
        b_indices = []
        
        # Find all indices where substring a occurs
        for i in range(len_s - len_a + 1):
            if s[i:i + len_a] == a:
                a_indices.append(i)
        
        # Find all indices where substring b occurs
        for j in range(len_s - len_b + 1):
            if s[j:j + len_b] == b:
                b_indices.append(j)
        
        beautiful_indices = []
        
        # Check for each index in a_indices if there exists a valid index in b_indices
        for i in a_indices:
            for j in b_indices:
                if abs(i - j) <= k:
                    beautiful_indices.append(i)
                    break
        
        return beautiful_indices