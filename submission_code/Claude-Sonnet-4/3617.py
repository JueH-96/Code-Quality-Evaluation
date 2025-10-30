class Solution:
    def possibleStringCount(self, word: str) -> int:
        if len(word) == 1:
            return 1
        
        # Find groups of consecutive identical characters
        groups = []
        i = 0
        while i < len(word):
            j = i
            while j < len(word) and word[j] == word[i]:
                j += 1
            groups.append(j - i)  # length of this group
            i = j
        
        # Count possibilities
        # 1 for no mistake case
        result = 1
        
        # For each group with length > 1, add (length - 1) possibilities
        for group_len in groups:
            if group_len > 1:
                result += group_len - 1
        
        return result