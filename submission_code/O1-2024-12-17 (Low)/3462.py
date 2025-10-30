class Solution:
    def doesAliceWin(self, s: str) -> bool:
        # Count the number of vowels in s
        vowels = {'a', 'e', 'i', 'o', 'u'}
        vowel_count = sum(ch in vowels for ch in s)
        
        # If there are fewer than 2 vowels, Alice cannot force a win
        #   - 0 vowels => Alice cannot remove an odd-vowel substring at all
        #   - 1 vowel  => after Alice removes that 1-vowel substring, Bob
        #                 can remove all consonants (which have 0 vowels),
        #                 leaving nothing for Alice.
        # Otherwise, if there are at least 2 vowels, Alice can always
        # maneuver to win by leaving Bob in a losing position.
        return vowel_count >= 2