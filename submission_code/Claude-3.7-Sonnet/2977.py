class Solution:
    def isAcronym(self, words: List[str], s: str) -> bool:
        # First check if the lengths match - an acronym must have the same 
        # length as the number of words
        if len(s) != len(words):
            return False
        
        # Check if each character in s matches the first character of each word
        for i, word in enumerate(words):
            if s[i] != word[0]:
                return False
        
        return True