class Solution:
    def doesAliceWin(self, s: str) -> bool:
        vowels = set('aeiou')
        # Alice can move if and only if there's at least one vowel in s.
        # If there is no vowel, Alice has no valid odd‐vowel substring to remove and loses.
        return any(ch in vowels for ch in s)