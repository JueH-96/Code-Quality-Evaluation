class Solution:
    def doesAliceWin(self, s: str) -> bool:
        vowels = {'a', 'e', 'i', 'o', 'u'}
        n = len(s)
        total_vowels = sum(1 for char in s if char in vowels)

        if total_vowels % 2 == 1:
            return True
        else:
            # Check if Alice has a valid move
            # Alice has a valid move if there exists a substring with an odd number of vowels

            # Iterate through all possible substrings
            for i in range(n):
                for j in range(i, n):
                    substring = s[i : j + 1]
                    vowel_count = sum(1 for char in substring if char in vowels)
                    if vowel_count % 2 == 1:
                        return True  # Alice can make a move

            return False  # Alice cannot make a move