class Solution:
    def beautifulIndices(self, s: str, a: str, b: str, k: int) -> List[int]:
        # Find all occurrences of string a
        a_indices = []
        for i in range(len(s) - len(a) + 1):
            if s[i:i + len(a)] == a:
                a_indices.append(i)
        
        # Find all occurrences of string b
        b_indices = []
        for i in range(len(s) - len(b) + 1):
            if s[i:i + len(b)] == b:
                b_indices.append(i)
        
        # If no b indices, no beautiful indices possible
        if not b_indices:
            return []
        
        # Check which a indices are beautiful
        beautiful = []
        for i in a_indices:
            # Check if there exists any j in b_indices such that |j - i| <= k
            for j in b_indices:
                if abs(j - i) <= k:
                    beautiful.append(i)
                    break
        
        return beautiful