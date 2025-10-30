class Solution:
    def numberOfSpecialChars(self, word: str) -> int:
        lower_set = set()
        upper_set = set()
        
        for char in word:
            if char.islower():
                lower_set.add(char)
            else:
                # convert uppercase to lowercase
                upper_set.add(char.lower())
        
        # special letters appear in both sets
        return len(lower_set & upper_set)