class Solution:
    def doesAliceWin(self, s: str) -> bool:
        vowels = set('aeiou')
        
        def count_vowels(substr):
            return sum(1 for c in substr if c in vowels)
            
        def has_valid_move(s, is_alice):
            n = len(s)
            # Try all possible substrings
            for i in range(n):
                for j in range(i + 1, n + 1):
                    substr = s[i:j]
                    vowel_count = count_vowels(substr)
                    # For Alice's turn, need odd number of vowels
                    # For Bob's turn, need even number of vowels
                    if (is_alice and vowel_count % 2 == 1) or (not is_alice and vowel_count % 2 == 0):
                        return True
            return False
            
        def can_win(s, is_alice, memo=None):
            if memo is None:
                memo = {}
                
            if s in memo:
                return memo[s]
                
            # Base case - if current player has no valid moves, they lose
            if not has_valid_move(s, is_alice):
                memo[s] = False
                return False
                
            n = len(s)
            # Try all possible substrings
            for i in range(n):
                for j in range(i + 1, n + 1):
                    substr = s[i:j]
                    vowel_count = count_vowels(substr)
                    
                    # Check if this is a valid move for current player
                    if (is_alice and vowel_count % 2 == 1) or (not is_alice and vowel_count % 2 == 0):
                        # Make the move and check opponent's result
                        new_s = s[:i] + s[j:]
                        # If opponent loses after our move, we win
                        if not can_win(new_s, not is_alice, memo):
                            memo[s] = True
                            return True
                            
            # If no winning move found, we lose
            memo[s] = False
            return False
            
        return can_win(s, True)