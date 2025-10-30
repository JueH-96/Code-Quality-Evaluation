class Solution:
    def minimumLength(self, s: str) -> int:
        from collections import defaultdict
        
        # Dictionary to store the indices of each character
        char_indices = defaultdict(list)
        for index, char in enumerate(s):
            char_indices[char].append(index)
        
        # Set to store the indices that are removed
        removed_indices = set()
        
        # Function to remove elements around a given index
        def remove_elements(char, idx):
            indices = char_indices[char]
            left_idx = right_idx = None
            
            # Find the closest left index
            for i in reversed(range(len(indices))):
                if indices[i] < idx:
                    left_idx = indices[i]
                    break
            
            # Find the closest right index
            for i in range(len(indices)):
                if indices[i] > idx:
                    right_idx = indices[i]
                    break
            
            # Remove the indices if valid
            if left_idx is not None and right_idx is not None:
                removed_indices.add(left_idx)
                removed_indices.add(right_idx)
                return True
            return False
        
        # Iterate over each character's indices
        for char, indices in char_indices.items():
            if len(indices) < 3:
                continue  # We need at least three indices to perform any removal
            
            # Try to remove elements around each index (except the first and last)
            for i in range(1, len(indices) - 1):
                if indices[i] not in removed_indices:
                    remove_elements(char, indices[i])
        
        # Calculate the length of the string after removals
        final_length = len(s) - len(removed_indices)
        return final_length