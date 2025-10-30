class Solution:
    def calculateScore(self, s: str) -> int:
        # Create a mapping from each letter to its mirror letter
        alphabet = "abcdefghijklmnopqrstuvwxyz"
        mirror_map = {ch: alphabet[25 - i] for i, ch in enumerate(alphabet)}
        
        # Dictionary to store the last unmarked index of each letter
        last_unmarked_index = {}
        
        # Set to keep track of marked indices
        marked_indices = set()
        
        # Initialize score
        score = 0
        
        # Iterate over the string
        for i, char in enumerate(s):
            mirror_char = mirror_map[char]
            
            # Check if there's an unmarked index for the mirror character
            if mirror_char in last_unmarked_index:
                j = last_unmarked_index[mirror_char]
                
                # Ensure j is unmarked
                if j not in marked_indices:
                    # Calculate score increment
                    score += i - j
                    
                    # Mark the indices
                    marked_indices.add(j)
                    marked_indices.add(i)
                    
                    # Remove the index from last_unmarked_index as it's now marked
                    del last_unmarked_index[mirror_char]
            
            # If current index is not marked, update the last unmarked index for this character
            if i not in marked_indices:
                last_unmarked_index[char] = i
        
        return score