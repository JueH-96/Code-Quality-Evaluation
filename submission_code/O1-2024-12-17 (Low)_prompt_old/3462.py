class Solution:
    def doesAliceWin(self, s: str) -> bool:
        # If s has no vowels at all, Alice cannot remove any odd-vowel substring (there are none), so she loses immediately.
        # If there is at least one vowel in s, there is at least one substring with an odd number of vowels.
        # In that case, Alice (playing first and optimally) can always force a win.
        vowels = {'a', 'e', 'i', 'o', 'u'}
        for ch in s:
            if ch in vowels:
                return True
        return False