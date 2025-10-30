class Solution:
    def beautifulIndices(self, s: str, a: str, b: str, k: int) -> List[int]:
        # Find all occurrences of pattern a
        a_indices = []
        for i in range(len(s) - len(a) + 1):
            if s[i:i+len(a)] == a:
                a_indices.append(i)
                
        # Find all occurrences of pattern b
        b_indices = []
        for i in range(len(s) - len(b) + 1):
            if s[i:i+len(b)] == b:
                b_indices.append(i)
        
        # For each a_index, check if there exists a b_index within k distance
        result = []
        b_idx = 0
        
        for a_idx in a_indices:
            # Move b_idx forward until we find a potential match
            while b_idx < len(b_indices) and b_indices[b_idx] < a_idx - k:
                b_idx += 1
                
            # Check if current b_idx is within k distance
            if b_idx < len(b_indices) and abs(a_idx - b_indices[b_idx]) <= k:
                result.append(a_idx)
            else:
                # Check if any remaining b_indices are within k distance
                j = b_idx
                while j < len(b_indices) and b_indices[j] <= a_idx + k:
                    if abs(a_idx - b_indices[j]) <= k:
                        result.append(a_idx)
                        break
                    j += 1
                    
        return result