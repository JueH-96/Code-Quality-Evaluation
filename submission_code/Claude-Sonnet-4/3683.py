class Solution:
    def answerString(self, word: str, numFriends: int) -> str:
        if numFriends == 1:
            return word
        
        n = len(word)
        all_substrings = set()
        
        def generate_splits(start_idx, parts_left, current_split):
            if parts_left == 1:
                # Last part takes the remaining string
                current_split.append(word[start_idx:])
                for part in current_split:
                    all_substrings.add(part)
                current_split.pop()
                return
            
            # Try all possible lengths for the current part
            # Need at least (parts_left - 1) characters for remaining parts
            for end_idx in range(start_idx + 1, n - parts_left + 2):
                current_split.append(word[start_idx:end_idx])
                generate_splits(end_idx, parts_left - 1, current_split)
                current_split.pop()
        
        generate_splits(0, numFriends, [])
        return max(all_substrings)