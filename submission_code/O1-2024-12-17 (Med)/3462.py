class Solution:
    def doesAliceWin(self, s: str) -> bool:
        """
        Observation/Insight:

        1) If there are no vowels in s, then Alice cannot find a substring with an odd number of vowels
           (since that would be 0, which is even). Alice cannot make a move, so she loses immediately.

        2) If there is exactly one vowel in s, Alice can remove that single-vowel substring (odd=1).
           The remaining string has zero vowels (all consonants). On Bob's turn, a substring with
           zero vowels is allowed (0 is even), so Bob can remove whatever remains. Alice then has
           no moves on the empty string, so Alice loses.

        3) If there are at least two vowels in s, Alice always has a winning strategy. Examples:
           - "aa": Alice removes one "a" → leftover is "a" → Bob has no even-vowel substring.
           - "abba": Alice can remove "abb" (1 vowel) → leftover "a" → Bob cannot move.
           - "leetcoder" (the given example with 4 vowels) also shows a winning path for Alice.

        Thus, we only need to count how many vowels appear in s.
         - If the total vowel count is 0 or 1, return False (Alice loses).
         - Otherwise, return True (Alice wins).
        """

        vowels = set('aeiou')
        # Count vowels in s
        vowel_count = sum(ch in vowels for ch in s)

        # If <=1 vowel, Alice cannot win under optimal play
        return vowel_count > 1