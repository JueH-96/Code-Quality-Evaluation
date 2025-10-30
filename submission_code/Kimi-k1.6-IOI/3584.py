import bisect
from typing import List

class Solution:
    def validSequence(self, word1: str, word2: str) -> List[int]:
        m = len(word2)
        n = len(word1)
        if m == 0:
            return []
        
        # Preprocess character indices in word1
        char_indices = {}
        for idx, c in enumerate(word1):
            if c not in char_indices:
                char_indices[c] = []
            char_indices[c].append(idx)
        
        # Initialize DP table with (index, prev_j, prev_k)
        INF = float('inf')
        dp = [ [ (INF, -1, -1) for _ in range(2) ] for _ in range(m) ]
        
        # Initialize for j=0
        first_char = word2[0]
        if first_char in char_indices:
            indices = char_indices[first_char]
            first_idx = indices[0]
            dp[0][0] = (first_idx, -1, -1)
        dp[0][1] = (0, -1, -1)  # Always possible to take index 0 with one change
        
        # Process each j from 0 to m-2
        for j in range(m - 1):
            for k in range(2):
                current_idx, prev_j, prev_k = dp[j][k]
                if current_idx == INF:
                    continue
                
                next_char = word2[j + 1]
                
                # Option 1: No change, find next character in word1 after current_idx
                if next_char in char_indices:
                    indices = char_indices[next_char]
                    pos = bisect.bisect_right(indices, current_idx)
                    if pos < len(indices):
                        new_idx = indices[pos]
                        if new_idx < dp[j + 1][k][0]:
                            dp[j + 1][k] = (new_idx, j, k)
                
                # Option 2: Change the current character (only if k < 1)
                if k == 0:
                    next_idx = current_idx + 1
                    if next_idx < n:
                        if next_idx < dp[j + 1][1][0]:
                            dp[j + 1][1] = (next_idx, j, k)
        
        # Determine the best state at the last position
        best_idx = INF
        best_k = -1
        for k in range(2):
            if dp[m - 1][k][0] < best_idx:
                best_idx = dp[m - 1][k][0]
                best_k = k
        
        if best_k == -1:
            return []
        
        # Reconstruct the sequence
        sequence = []
        current_j, current_k = m - 1, best_k
        while current_j != -1:
            sequence.append(dp[current_j][current_k][0])
            prev_j, prev_k = dp[current_j][current_k][1], dp[current_j][current_k][2]
            current_j, current_k = prev_j, prev_k
        
        # Reverse to get the correct order
        sequence.reverse()
        return sequence