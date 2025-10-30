class Solution:
    def beautifulIndices(self, s: str, a: str, b: str, k: int) -> List[int]:
        # Find all starting indices for a in s
        a_len = len(a)
        a_indices = [i for i in range(len(s) - a_len + 1) if s[i:i+a_len] == a]
        
        # Find all starting indices for b in s
        b_len = len(b)
        b_indices = [j for j in range(len(s) - b_len + 1) if s[j:j+b_len] == b]
        
        # Use two-pointer approach to find beautiful indices
        result = []
        p_b = 0
        for i in a_indices:
            # Move p_b until b_indices[p_b] >= i - k
            while p_b < len(b_indices) and b_indices[p_b] < i - k:
                p_b += 1
            # Now, check if b_indices[p_b] <= i + k
            if p_b < len(b_indices) and b_indices[p_b] <= i + k:
                result.append(i)
        
        return result