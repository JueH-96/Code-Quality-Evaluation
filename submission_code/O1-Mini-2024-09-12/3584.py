from bisect import bisect_right
from typing import List

class Solution:
    def validSequence(self, word1: str, word2: str) -> List[int]:
        from bisect import bisect_right

        # Preprocess: mapping from character to sorted list of indices in word1
        from collections import defaultdict
        char_indices = defaultdict(list)
        for idx, char in enumerate(word1):
            char_indices[char].append(idx)
        
        n, m = len(word1), len(word2)
        if m > n:
            return []
        
        # Initialize sequences: zero_mismatch and one_mismatch
        zero_mismatch = []  # list of indices
        one_mismatch = []   # list of indices
        
        # Initialize previous index for zero_mismatch
        prev_zero = -1
        # Try to build zero_mismatch sequence
        for i in range(m):
            char = word2[i]
            if char not in char_indices:
                zero_mismatch = None
                break
            lst = char_indices[char]
            pos = bisect_right(lst, prev_zero)
            if pos == len(lst):
                zero_mismatch = None
                break
            index = lst[pos]
            zero_mismatch.append(index)
            prev_zero = index
        
        # If zero_mismatch exists, set initial candidate
        candidates = []
        if zero_mismatch:
            candidates.append(zero_mismatch)
        
        # Now, try sequences with one mismatch
        # We will iterate through each position to allow a mismatch there
        # and attempt to build the smallest possible sequence
        # Initialize prefix arrays
        prefix = [None] * m
        prev = -1
        for i in range(m):
            char = word2[i]
            if char not in char_indices:
                break
            lst = char_indices[char]
            pos = bisect_right(lst, prev)
            if pos == len(lst):
                break
            index = lst[pos]
            prefix[i] = index
            prev = index
        else:
            # Complete exact match already handled
            pass
        
        # Compute suffix arrays
        suffix = [None] * m
        prev = len(word1)
        for i in range(m-1, -1, -1):
            char = word2[i]
            if char not in char_indices:
                break
            lst = char_indices[char]
            pos = bisect_right(lst, prev - 1) - 1
            if pos < 0:
                break
            index = lst[pos]
            suffix[i] = index
            prev = index
        # Now, try to allow mismatch at each position
        for i in range(m):
            # Find prefix up to i-1
            if i == 0:
                prev_index = -1
            else:
                if prefix[i-1] is None:
                    continue
                prev_index = prefix[i-1]
            # Choose any character at position i: choose the smallest possible index > prev_index
            # which is the earliest possible index
            candidate_indices = []
            # All indices after prev_index
            # The smallest possible index is prev_index +1
            choice = prev_index + 1
            # Find the smallest index in word1 >= choice
            # Since word1 consists of lowercase letters, the smallest index overall is the smallest index >= choice
            # Iterate through all characters to find the smallest index >= choice
            # However, to make it efficient, we iterate through word1 from choice to end and pick the first index
            # This is equivalent to choosing the smallest possible index >= choice
            # Which is choice itself if it's within bounds
            if choice < n:
                mismatch_index = choice
                # Now, build the sequence
                seq = []
                # Add prefix up to i-1
                if i > 0:
                    seq.extend(prefix[:i])
                # Add mismatch index
                seq.append(mismatch_index)
                # Now, match the rest from i+1 to m-1
                valid = True
                current = mismatch_index
                for j in range(i+1, m):
                    char = word2[j]
                    if char not in char_indices:
                        valid = False
                        break
                    lst = char_indices[char]
                    pos = bisect_right(lst, current)
                    if pos == len(lst):
                        valid = False
                        break
                    current = lst[pos]
                    seq.append(current)
                if valid:
                    candidates.append(seq)
        # Now, find the lex smallest sequence among candidates
        if not candidates:
            return []
        # Compare sequences lex order
        min_seq = None
        for seq in candidates:
            if min_seq is None or seq < min_seq:
                min_seq = seq
        return min_seq