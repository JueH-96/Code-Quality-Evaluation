class Solution:
    def beautifulIndices(self, s: str, a: str, b: str, k: int) -> List[int]:
        n = len(s)
        a_len, b_len = len(a), len(b)
        a_indices = [i for i in range(n - a_len + 1) if s[i:i + a_len] == a]
        b_indices = [i for i in range(n - b_len + 1) if s[i:i + b_len] == b]
        
        beautiful_indices = []
        a_ptr, b_ptr = 0, 0
        
        while a_ptr < len(a_indices) and b_ptr < len(b_indices):
            a_idx, b_idx = a_indices[a_ptr], b_indices[b_ptr]
            if abs(a_idx - b_idx) <= k:
                beautiful_indices.append(a_idx)
                a_ptr += 1
                b_ptr += 1
            elif a_idx < b_idx:
                a_ptr += 1
            else:
                b_ptr += 1
        
        return beautiful_indices