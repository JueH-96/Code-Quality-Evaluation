class Solution:
    def validSequence(self, word1: str, word2: str) -> list:
        m = len(word2)
        n = len(word1)
        if m == 0:
            return []
        
        # Part a: Check for a sequence with zero changes
        sequence = []
        i, j = 0, 0
        while i < n and j < m:
            if word1[i] == word2[j]:
                sequence.append(i)
                j += 1
            i += 1
        if j == m:
            return sequence
        
        # If part a failed, proceed to part b
        best_sequence = None
        
        # Precompute for each character in word1, the indices where it appears
        from collections import defaultdict
        char_indices = defaultdict(list)
        for idx, c in enumerate(word1):
            char_indices[c].append(idx)
        
        for k in range(m):
            # Step 1: Find the first part (0 to k-1)
            first_part = []
            i1 = 0
            j1 = 0
            while j1 < k and i1 < n:
                if word1[i1] == word2[j1]:
                    first_part.append(i1)
                    j1 += 1
                i1 += 1
            if j1 < k:
                continue  # Can't form the first part
            
            # Step 2: Find i_k (earliest possible >= i1)
            i_k = i1
            if i_k >= n:
                continue
            
            # Step 3: Find the remaining parts (k+1 to m-1)
            i2 = i_k + 1
            j2 = k + 1
            remaining = []
            while j2 < m and i2 < n:
                if word1[i2] == word2[j2]:
                    remaining.append(i2)
                    j2 += 1
                i2 += 1
            if j2 < m:
                continue  # Can't form the remaining part
            
            # Combine the parts
            full_sequence = first_part + [i_k] + remaining
            if len(full_sequence) != m:
                continue  # This should not happen as we checked above
            
            # Ensure the sequence is strictly increasing
            if all(full_sequence[i] < full_sequence[i+1] for i in range(len(full_sequence)-1)):
                if (best_sequence is None) or (full_sequence < best_sequence):
                    best_sequence = full_sequence
        
        if best_sequence is not None:
            return best_sequence
        else:
            return []