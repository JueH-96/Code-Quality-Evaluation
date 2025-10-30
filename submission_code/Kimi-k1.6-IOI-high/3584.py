import bisect
from collections import defaultdict
from typing import List

class Solution:
    def validSequence(self, word1: str, word2: str) -> List[int]:
        m = len(word2)
        n = len(word1)
        if m > n:
            return []
        
        # Preprocess character indices
        char_indices = defaultdict(list)
        for idx, c in enumerate(word1):
            char_indices[c].append(idx)
        
        # Preprocess next_not_c for each character
        next_not_c = defaultdict(list)
        for c in 'abcdefghijklmnopqrstuvwxyz':
            nn = [-1] * n  # Initialize all to -1, meaning no such index found
            last_non_c = -1  # Tracks the last known non-c position starting from the end
            for j in range(n-1, -1, -1):
                if word1[j] != c:
                    last_non_c = j
                    nn[j] = j
                else:
                    nn[j] = last_non_c  # This will be >=j if found, else -1
            next_not_c[c] = nn
        
        prev_a = None  # Best sequence with 0 mismatches (up to i steps)
        prev_b = None  # Best sequence with up to 1 mismatch (up to i steps)
        
        for i in range(m):
            current_char = word2[i]
            new_a = None
            new_b_candidates = []
            
            # Compute new_a (exact match for current_char)
            if i == 0:
                indices = char_indices.get(current_char, [])
                if indices:
                    new_a = [indices[0]]
            else:
                # Check if prev_a is valid (length exactly i)
                if prev_a and len(prev_a) == i:
                    prev_index = prev_a[-1]
                    indices = char_indices.get(current_char, [])
                    idx = bisect.bisect_right(indices, prev_index)
                    if idx < len(indices):
                        new_a = list(prev_a)
                        new_a.append(indices[idx])
            
            # Compute new_b candidates
            # Candidate 1: from prev_a (add a mismatch)
            if prev_a and len(prev_a) == i:
                prev_index = prev_a[-1]
                j = prev_index + 1
                if j < n:
                    c = current_char
                    next_j = next_not_c[c][j]
                    if next_j != -1:
                        candidate = list(prev_a)
                        candidate.append(next_j)
                        new_b_candidates.append(candidate)
            
            # Candidate 2: from prev_b (add exact match)
            if prev_b and len(prev_b) == i:
                prev_index = prev_b[-1]
                indices = char_indices.get(current_char, [])
                idx = bisect.bisect_right(indices, prev_index)
                if idx < len(indices):
                    candidate = list(prev_b)
                    candidate.append(indices[idx])
                    new_b_candidates.append(candidate)
            
            # Handle case where no candidates for i=0
            if i == 0 and not new_b_candidates:
                c = current_char
                next_j = next_not_c[c][0]
                if next_j != -1:
                    new_b_candidates.append([next_j])
            
            # Determine new_b
            new_b = None
            if new_b_candidates:
                # Sort to find lex smallest
                new_b_candidates.sort()
                new_b = new_b_candidates[0]
            
            # Update previous sequences
            # Update prev_a if new_a is valid and has correct length
            if new_a is not None and len(new_a) == i + 1:
                prev_a = new_a
            else:
                prev_a = None
            
            # Update prev_b
            prev_b = new_b
        
        # Collect results and return the lex smallest
        result = []
        if prev_a and len(prev_a) == m:
            result.append(prev_a)
        if prev_b and len(prev_b) == m:
            result.append(prev_b)
        
        if not result:
            return []
        # Sort the results and pick the lex smallest
        result.sort()
        return result[0]