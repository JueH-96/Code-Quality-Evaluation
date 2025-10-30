class Solution:
    def doesAliceWin(self, s: str) -> bool:
        """
        Alice can make a move iff the string contains at least one vowel
        (she can always delete that single–letter substring which has 1 – an odd
        – number of vowels).

        If the string has no vowel, Alice has no legal move and loses instantly.
        Otherwise Alice can always force a win:

        • If the total number of vowels is odd she deletes the whole string and the
          game ends immediately – Bob has no move.

        • If the total number of vowels is even she deletes any single vowel
          (an odd-vowel substring).  
          The remaining string now has an odd number of vowels.
          Bob is required to delete an even-vowel substring, so he cannot
          delete the whole (odd-vowel) string; at least one vowel survives.
          On Alice’s next turn the string still has an odd number of vowels,
          so she removes the whole remaining string and wins.

        Hence Alice loses only when the initial string contains zero vowels.
        """

        vowels = set('aeiou')
        for ch in s:
            if ch in vowels:          # at least one vowel exists
                return True           # Alice wins
        return False                  # no vowel -> Alice cannot move