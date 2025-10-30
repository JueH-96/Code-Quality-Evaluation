class Solution:
    def doesAliceWin(self, s: str) -> bool:
        vowels = "aeiou"
        n = len(s)
        
        def count_vowels(sub):
            count = 0
            for char in sub:
                if char in vowels:
                    count += 1
            return count
        
        
        def can_win(current_s, is_alice_turn):
            if not current_s:
                return not is_alice_turn
            
            
            for i in range(len(current_s)):
                for j in range(i, len(current_s)):
                    sub = current_s[i:j+1]
                    vowel_count = count_vowels(sub)
                    
                    if is_alice_turn and vowel_count % 2 != 0:
                        if not can_win(current_s[:i] + current_s[j+1:], False):
                            return True
                    elif not is_alice_turn and vowel_count % 2 == 0:
                        if not can_win(current_s[:i] + current_s[j+1:], True):
                            return True
            
            return False
        
        return can_win(s, True)