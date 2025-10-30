class Solution:
    def doesAliceWin(self, s: str) -> bool:
        # Alice can only make a move if there is at least one vowel in the string.
        # If there are no vowels, Alice cannot remove any "odd-vowel" substring and loses immediately.
        # Otherwise, with at least one vowel present, Alice always has a winning strategy.
        vowels = set('aeiou')
        for ch in s:
            if ch in vowels:
                return True
        return False