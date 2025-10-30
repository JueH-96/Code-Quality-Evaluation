class Solution:
    def doesAliceWin(self, s: str) -> bool:
        # Define the set of English vowels
        vowel_chars = "aeiou"
        
        # Count the number of vowels in the input string s
        vowel_count = 0
        for char in s:
            if char in vowel_chars:
                vowel_count += 1
        
        # Analyze the game's winning condition:
        # Alice must remove a substring with an odd number of vowels.
        # Bob must remove a substring with an even number of vowels.
        # The first player who cannot make a move loses.
        
        # If the initial string s contains 0 vowels:
        # Any non-empty substring of s will also contain 0 vowels, which is an even number.
        # Alice needs to remove a substring with an odd number of vowels.
        # Since no such substring exists, Alice cannot make a move on her first turn.
        # Therefore, if vowel_count is 0, Alice loses immediately.
        
        # If the initial string s contains at least one vowel (vowel_count > 0):
        # Alice needs to remove a substring with an odd number of vowels.
        # Since there is at least one vowel in s, Alice can choose a substring
        # consisting of a single vowel character. This substring is non-empty
        # and contains exactly 1 vowel, which is an odd number.
        # Thus, Alice can always make a valid move on her first turn if vowel_count > 0.
        #
        # In impartial games like this, if the starting player is not in a losing
        # state, they typically have a winning strategy, assuming optimal play.
        # The only initial losing state for Alice is when she cannot make her first move,
        # which occurs when there are no vowels.
        # If she can make a move (i.e., vowel_count > 0), she wins.
        
        return vowel_count > 0