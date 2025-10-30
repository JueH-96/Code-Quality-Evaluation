class Solution:
    def numberOfSpecialChars(self, word: str) -> int:
        lower_set = set()
        upper_set = set()
        
        for ch in word:
            if ch.islower():
                lower_set.add(ch)
            else:
                upper_set.add(ch)
        
        # Convert uppercase letters to lowercase and intersect with lower_set
        # The result is the count of special letters.
        special_chars = lower_set.intersection({c.lower() for c in upper_set})
        return len(special_chars)