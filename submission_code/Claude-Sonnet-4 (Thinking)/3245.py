class Solution:
    def beautifulIndices(self, s: str, a: str, b: str, k: int) -> List[int]:
        # Find all indices where a occurs
        a_indices = []
        for i in range(len(s) - len(a) + 1):
            if s[i:i+len(a)] == a:
                a_indices.append(i)
        
        # Find all indices where b occurs
        b_indices = []
        for i in range(len(s) - len(b) + 1):
            if s[i:i+len(b)] == b:
                b_indices.append(i)
        
        # For each a_index, check if there's a b_index within distance k
        beautiful = []
        for a_idx in a_indices:
            for b_idx in b_indices:
                if abs(b_idx - a_idx) <= k:
                    beautiful.append(a_idx)
                    break  # Found one, no need to check others
        
        return beautiful