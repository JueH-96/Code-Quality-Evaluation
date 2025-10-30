class Solution:
    def beautifulIndices(self, s: str, a: str, b: str, k: int) -> List[int]:
        beautiful_indices = []
        a_len = len(a)
        b_len = len(b)
        s_len = len(s)
        
        # Find all starting indices of substring a
        a_indices = [i for i in range(s_len - a_len + 1) if s[i:i + a_len] == a]
        
        # Find all starting indices of substring b
        b_indices = [i for i in range(s_len - b_len + 1) if s[i:i + b_len] == b]
        
        # Check each index of a to see if it is beautiful
        for i in a_indices:
            for j in b_indices:
                if abs(j - i) <= k:
                    beautiful_indices.append(i)
                    break  # No need to check other j's once a beautiful index is found
        
        return beautiful_indices