class Solution:
    def numberOfSpecialChars(self, word: str) -> int:
        
        # Set to store unique lowercase letters encountered in the word.
        # Example: if 'a' is in word, 'a' will be added to this set.
        seen_lowercase = set()
        
        # Set to store unique uppercase letters encountered in the word,
        # normalized to their lowercase form.
        # Example: if 'A' is in word, 'a' (from 'A'.lower()) will be added to this set.
        seen_uppercase_as_lowercase = set()
        
        for char in word:
            if char.islower():
                seen_lowercase.add(char)
            elif char.isupper():
                seen_uppercase_as_lowercase.add(char.lower())
                
        # A letter (e.g., 'x') is special if its lowercase form ('x') was seen
        # (i.e., 'x' is in seen_lowercase) AND its uppercase form ('X') was also seen
        # (which means 'x' would be in seen_uppercase_as_lowercase).
        # The intersection of these two sets will contain all such lowercase letters 'x'
        # that represent special letters.
        
        special_letters = seen_lowercase.intersection(seen_uppercase_as_lowercase)
        
        # The number of special letters is the number of elements in the intersection set.
        return len(special_letters)