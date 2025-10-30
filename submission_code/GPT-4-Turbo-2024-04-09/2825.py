class Solution:
    def minimizedStringLength(self, s: str) -> int:
        from collections import deque
        
        # Dictionary to store the indices of each character
        char_indices = {}
        
        for i, char in enumerate(s):
            if char not in char_indices:
                char_indices[char] = deque()
            char_indices[char].append(i)
        
        # Set to keep track of removed indices
        removed_indices = set()
        
        # Function to remove closest occurrences
        def remove_closest_occurrences(index, char):
            indices = char_indices[char]
            # Remove the current index
            indices.remove(index)
            # Remove closest left occurrence
            left_index = None
            right_index = None
            for idx in reversed(indices):
                if idx < index:
                    left_index = idx
                    break
            # Remove closest right occurrence
            for idx in indices:
                if idx > index:
                    right_index = idx
                    break
            if left_index is not None:
                indices.remove(left_index)
                removed_indices.add(left_index)
            if right_index is not None:
                indices.remove(right_index)
                removed_indices.add(right_index)
        
        # Iterate over each character and try to minimize the string
        for i, char in enumerate(s):
            if i in removed_indices:
                continue
            if char in char_indices and len(char_indices[char]) > 1:
                remove_closest_occurrences(i, char)
        
        # Calculate the length of the minimized string
        minimized_length = len(s) - len(removed_indices)
        return minimized_length