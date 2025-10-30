from typing import List

class Solution:
    def validSequence(self, word1: str, word2: str) -> List[int]:
        n, m = len(word1), len(word2)
        if m == 0:
            return []
        
        # Precompute next occurrence for each character
        next_pos = [{} for _ in range(n+1)]
        for i in range(n-1, -1, -1):
            next_pos[i] = next_pos[i+1].copy()
            next_pos[i][word1[i]] = i
        
        # Function to find exact subsequence indices
        def get_exact_indices(s):
            indices = []
            current = -1
            for c in s:
                current += 1
                if current >= n:
                    return None
                pos = next_pos[current].get(c, None)
                if pos is None:
                    return None
                indices.append(pos)
                current = pos
            return indices
        
        # Check exact match
        exact = get_exact_indices(word2)
        if exact is not None:
            return exact
        
        # Iterate each possible k to allow mismatch
        for k in range(m):
            # Find prefix_indices for word2[0..k-1]
            prefix = []
            current = -1
            valid = True
            for i in range(k):
                c = word2[i]
                current += 1
                if current >= n:
                    valid = False
                    break
                pos = next_pos[current].get(c, None)
                if pos is None:
                    valid = False
                    break
                prefix.append(pos)
                current = pos
            if not valid:
                continue
            
            # Find idx_k
            idx_k = current + 1 if k > 0 else 0
            if idx_k >= n:
                continue
            
            # Find suffix_indices for word2[k+1..m-1] starting from idx_k + 1
            suffix = []
            current = idx_k
            valid_suffix = True
            for i in range(k+1, m):
                c = word2[i]
                current += 1
                if current >= n:
                    valid_suffix = False
                    break
                pos = next_pos[current].get(c, None)
                if pos is None:
                    valid_suffix = False
                    break
                suffix.append(pos)
                current = pos
            if valid_suffix:
                return prefix + [idx_k] + suffix
        
        return []