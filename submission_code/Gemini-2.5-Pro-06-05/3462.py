class Solution:
    def doesAliceWin(self, s: str) -> bool:
        """
        Determines if Alice wins the game on the given string.

        The game's outcome simplifies based on the presence of vowels.
        - Alice's move: Remove a substring with an odd number of vowels.
        - Bob's move: Remove a substring with an even number of vowels.
        - A player loses if they cannot make a move.

        Analysis:
        1. If the string `s` contains no vowels:
           Any substring will also have 0 vowels. 0 is an even number.
           Alice needs to remove a substring with an odd number of vowels, which is impossible.
           So, if there are no vowels, Alice loses on her first turn.

        2. If the string `s` contains at least one vowel:
           Alice has a winning strategy.
           - If the total number of vowels `V` is odd:
             Alice can remove the entire string `s` (a valid move since V is odd).
             The string becomes empty, Bob has no move, so Alice wins.
           - If the total number of vowels `V` is even and positive:
             Alice can remove a substring with an odd number of vowels (e.g., a single vowel).
             The remaining string will have `V - odd = even - odd = odd` vowels.
             It's now Bob's turn, facing a string with an odd number of vowels.
             Bob must remove a substring with an even number of vowels.
             The string he leaves for Alice will have `odd - even = odd` vowels.
             No matter what Bob does, he must leave a string with an odd number of vowels for Alice.
             When it's Alice's turn with an odd-vowel string, she can use the strategy above (remove the whole string) to win.

        Therefore, Alice wins if and only if the string contains at least one vowel.
        """
        
        vowels = {'a', 'e', 'i', 'o', 'u'}
        
        # We just need to check if any character in the string is a vowel.
        for char in s:
            if char in vowels:
                return True
        
        return False