class Solution:
    def doesAliceWin(self, s: str) -> bool:
        # Key observation:
        # A valid move for Alice is to remove a substring that contains an odd number of vowels.
        # Notice that a single vowel (like "a") is such a valid substring.
        # Thus, if the string s contains at least one vowel then on her very first move
        # Alice can always remove just that one vowel (or even more cleverly remove a substring
        # that ends the game immediately if possible).
        #
        # In contrast, if s contains no vowel then every substring will contain 0 vowels (an even number)
        # and so Alice would have no valid move at the start. In that case Bob wins by default.
        #
        # Therefore, under optimal play, Alice wins if and only if s has at least one vowel.
        #
        # (Note: While the complete game might look like a subtle substring‐deletion game,
        #  one may prove using invariants that if there is any possibility for a move by Alice,
        #  she can always force a win. The key “loophole” is that on a turn when s has an odd
        #  number of vowels, Alice can remove the entire string in one move, immediately winning.)
        vowels = set("aeiou")
        for ch in s:
            if ch in vowels:
                return True
        return False