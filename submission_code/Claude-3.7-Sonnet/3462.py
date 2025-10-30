class Solution:
    def doesAliceWin(self, s: str) -> bool:
        vowels = {'a', 'e', 'i', 'o', 'u'}
        memo = {}
        
        def can_win(current_s, is_alice_turn):
            # Base case: if string is empty, current player loses
            if not current_s:
                return False
            
            # Check if we've already computed this state
            if (current_s, is_alice_turn) in memo:
                return memo[(current_s, is_alice_turn)]
            
            # Determine which vowel parity the current player needs
            vowel_parity = 1 if is_alice_turn else 0  # Alice needs odd, Bob needs even
            
            # Quick win check: can the player remove the entire string?
            total_vowels = sum(1 for c in current_s if c in vowels)
            if total_vowels % 2 == vowel_parity:
                memo[(current_s, is_alice_turn)] = True
                return True
            
            # Try all possible substrings
            for i in range(len(current_s)):
                vowel_count = 0
                for j in range(i, len(current_s)):
                    if current_s[j] in vowels:
                        vowel_count += 1
                    
                    if vowel_count % 2 == vowel_parity:  # Valid move
                        new_s = current_s[:i] + current_s[j+1:]
                        # If opponent loses after this move, current player wins
                        if not can_win(new_s, not is_alice_turn):
                            memo[(current_s, is_alice_turn)] = True
                            return True
            
            # If no winning move found, current player loses
            memo[(current_s, is_alice_turn)] = False
            return False
        
        # Alice plays first
        return can_win(s, True)