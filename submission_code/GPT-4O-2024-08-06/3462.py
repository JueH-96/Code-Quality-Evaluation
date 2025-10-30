class Solution:
    def doesAliceWin(self, s: str) -> bool:
        # Function to count vowels in a string
        def count_vowels(substring):
            return sum(1 for char in substring if char in 'aeiou')

        # Total number of vowels in the string
        total_vowels = count_vowels(s)
        
        # If the total number of vowels is odd, Alice wins
        # because she can always remove a substring with an odd number of vowels
        # leaving Bob with an even number of vowels, which he cannot play with.
        # If the total number of vowels is even, Bob wins because Alice cannot
        # make the first move.
        return total_vowels % 2 == 1