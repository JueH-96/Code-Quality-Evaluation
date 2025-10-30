import bisect
from collections import defaultdict
from typing import List

class Solution:
    def validSequence(self, word1: str, word2: str) -> List[int]:
        pos = defaultdict(list)
        for i, c in enumerate(word1):
            pos[c].append(i)
        
        # Compute left_indices for the standard sequence
        left_indices = []
        j = 0
        for k in range(len(word2)):
            c = word2[k]
            if c not in pos:
                left_indices.extend([-1] * (len(word2) - k))
                break
            idx_list = pos[c]
            # Find the first occurrence >= j
            i_pos = bisect.bisect_left(idx_list, j)
            if i_pos == len(idx_list):
                left_indices.extend([-1] * (len(word2) - k))
                break
            current_idx = idx_list[i_pos]
            left_indices.append(current_idx)
            j = current_idx + 1
        
        # Check if standard sequence is valid
        if left_indices and left_indices[-1] != -1:
            return left_indices
        
        # Now check for substitution possibilities
        best_candidate = None
        for i in range(len(word2)):
            # Check if first i characters can be matched
            if i > 0 and left_indices[i-1] == -1:
                continue
            # Get prev_j
            if i == 0:
                prev_j = -1
            else:
                prev_j = left_indices[i-1]
            # Find x > prev_j where word1[x] != word2[i]
            start = prev_j + 1
            x = None
            # Iterate from start to len(word1)-1
            for j_x in range(start, len(word1)):
                if word1[j_x] != word2[i]:
                    x = j_x
                    break
            if x is None:
                continue
            # Check if remaining can be matched
            current_j = x + 1
            temp_indices = left_indices[0:i] if i > 0 else []
            temp_indices.append(x)
            valid = True
            for k in range(i+1, len(word2)):
                c = word2[k]
                if c not in pos:
                    valid = False
                    break
                idx_list = pos[c]
                i_pos = bisect.bisect_left(idx_list, current_j)
                if i_pos == len(idx_list):
                    valid = False
                    break
                current_idx = idx_list[i_pos]
                temp_indices.append(current_idx)
                current_j = current_idx + 1
            if valid and len(temp_indices) == len(word2):
                # Update best_candidate
                if best_candidate is None or temp_indices < best_candidate:
                    best_candidate = temp_indices
        
        return best_candidate if best_candidate is not None else []