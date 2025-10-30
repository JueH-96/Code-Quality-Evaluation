class Solution:
    def doesAliceWin(self, s: str) -> bool:
        # Define the vowels
        vowels = set('aeiou')
        
        # Initialize a 2D array to store the count of vowels in substrings
        n = len(s)
        vowel_count = [[0] * n for _ in range(n)]
        
        # Count the vowels in substrings
        for i in range(n):
            for j in range(i, n):
                vowel_count[i][j] = sum(1 for k in range(i, j + 1) if s[k] in vowels)
        
        # Function to check if Alice can win
        def can_alice_win(i, j):
            # If the substring has an odd number of vowels, Alice can win
            if vowel_count[i][j] % 2 == 1:
                return True
            # If the substring has an even number of vowels, Bob can win
            else:
                return False
        
        # Function to check if Bob can win
        def can_bob_win(i, j):
            # If the substring has an even number of vowels, Bob can win
            if vowel_count[i][j] % 2 == 0:
                return True
            # If the substring has an odd number of vowels, Alice can win
            else:
                return False
        
        # Check if Alice can win
        for i in range(n):
            for j in range(i, n):
                # If Alice can win in this substring, return True
                if can_alice_win(i, j):
                    return True
        
        # If Alice cannot win in any substring, return False
        return False