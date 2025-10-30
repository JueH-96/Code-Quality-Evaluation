from collections import defaultdict

class Solution:
    def lastNonEmptyString(self, s: str) -> str:
        # Track the indices for each character
        char_indices = defaultdict(list)
        for idx, char in enumerate(s):
            char_indices[char].append(idx)
        
        # Find the maximum occurrence count
        max_count = 0
        for indices in char_indices.values():
            current_length = len(indices)
            if current_length > max_count:
                max_count = current_length
        
        # Collect the indices of the max_count-th occurrence for each character with max_count occurrences
        selected_indices = []
        for indices in char_indices.values():
            if len(indices) == max_count:
                selected_indices.append(indices[max_count - 1])
        
        # Sort the indices to maintain the original order and build the result
        selected_indices.sort()
        return ''.join(s[i] for i in selected_indices)