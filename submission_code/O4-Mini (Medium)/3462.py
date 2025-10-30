class Solution:
    def doesAliceWin(self, s: str) -> bool:
        # Alice can make a move if and only if there is at least one vowel in the string.
        # If there's at least one vowel, Alice can remove a substring with an odd number of vowels
        # (for example a single vowel or the entire string if it has an odd vowel count),
        # and from that position she can force a win. If there are no vowels, Alice has no valid move.
        vowels = set('aeiou')
        for ch in s:
            if ch in vowels:
                return True
        return False