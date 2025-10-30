class Solution:
    def answerString(self, word: str, numFriends: int) -> str:
        def generate_splits(word, k, start, current_split, all_splits):
            # Base case: if we have k-1 splits, the last part is determined
            if k == 1:
                if start < len(word):
                    current_split.append(word[start:])
                    # Convert list of splits to tuple for hashable type
                    all_splits.add(tuple(current_split))
                    current_split.pop()
                return
            
            # Try different positions to split the word
            for i in range(start, len(word) - (k - 1)):
                # Take a substring and continue with remaining parts
                current_split.append(word[start:i + 1])
                generate_splits(word, k - 1, i + 1, current_split, all_splits)
                current_split.pop()
        
        # Store all unique splits
        all_splits = set()
        generate_splits(word, numFriends, 0, [], all_splits)
        
        # Collect all strings from all splits
        all_strings = set()
        for split in all_splits:
            for part in split:
                all_strings.add(part)
        
        # Return the lexicographically largest string
        if not all_strings:
            return ""
        return max(all_strings)