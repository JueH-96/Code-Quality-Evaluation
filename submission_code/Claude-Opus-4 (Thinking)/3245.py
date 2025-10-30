class Solution:
    def beautifulIndices(self, s: str, a: str, b: str, k: int) -> List[int]:
        # Find all occurrences of a in s
        a_indices = []
        for i in range(len(s) - len(a) + 1):
            if s[i:i+len(a)] == a:
                a_indices.append(i)
        
        # Find all occurrences of b in s
        b_indices = []
        for j in range(len(s) - len(b) + 1):
            if s[j:j+len(b)] == b:
                b_indices.append(j)
        
        # Find beautiful indices
        beautiful = []
        for i in a_indices:
            # Check if there exists a j in b_indices such that |j - i| <= k
            for j in b_indices:
                if abs(j - i) <= k:
                    beautiful.append(i)
                    break
        
        return beautiful