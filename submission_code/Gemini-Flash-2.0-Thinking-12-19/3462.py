class Solution:
    def doesAliceWin(self, s: str) -> bool:
        vowels = set(['a', 'e', 'i', 'o', 'u'])
        memo = {}
        
        def get_vowel_count(substring):
            count = 0
            for char in substring:
                if char in vowels:
                    count += 1
            return count
            
        def solve(current_s, is_alice_turn):
            if not current_s:
                return False # Current player cannot make a move, so loses
            if (current_s, is_alice_turn) in memo:
                return memo[(current_s, is_alice_turn)]
            
            possible_next_strings = set()
            can_move = False
            for i in range(len(current_s)):
                for j in range(i, len(current_s)):
                    substring = current_s[i:j+1]
                    vowel_count = get_vowel_count(substring)
                    if (is_alice_turn and vowel_count % 2 != 0) or (not is_alice_turn and vowel_count % 2 == 0):
                        can_move = True
                        next_s = current_s[:i] + current_s[j+1:]
                        possible_next_strings.add(next_s)
                        
            if not can_move:
                result = False # No valid move, current player loses
            else:
                win_possible = False
                for next_s in possible_next_strings:
                    if not solve(next_s, not is_alice_turn):
                        win_possible = True
                        break
                result = win_possible
                
            memo[(current_s, is_alice_turn)] = result
            return result
            
        return solve(s, True)