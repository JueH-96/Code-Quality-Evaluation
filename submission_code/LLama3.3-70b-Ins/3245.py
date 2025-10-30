from typing import List

class Solution:
    def beautifulIndices(self, s: str, a: str, b: str, k: int) -> List[int]:
        # Find all occurrences of string a in s
        a_indices = [i for i in range(len(s)) if s.startswith(a, i)]
        
        # Find all occurrences of string b in s
        b_indices = [i for i in range(len(s)) if s.startswith(b, i)]
        
        # Initialize an empty list to store beautiful indices
        beautiful_indices = []
        
        # Iterate over each index of string a
        for i in a_indices:
            # Check if there exists an index j of string b such that |j - i| <= k
            if any(abs(j - i) <= k for j in b_indices):
                beautiful_indices.append(i)
        
        # Return the list of beautiful indices in sorted order
        return sorted(set(beautiful_indices))