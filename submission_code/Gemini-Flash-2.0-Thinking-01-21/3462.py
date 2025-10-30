class Solution:
    def doesAliceWin(self, s: str) -> bool:
        vowels = set(['a', 'e', 'i', 'o', 'u'])
        memo = {}
        
        def count_vowel_substring(substring):
            count = 0
            for char in substring:
                if char in vowels:
                    count += 1
            return count
            
        def get_valid_moves(current_s, is_alice_turn):
            valid_substrings = []
            n = len(current_s)
            for i in range(n):
                for j in range(i, n):
                    substring = current_s[i:j+1]
                    vowel_count = count_vowel_substring(substring)
                    if is_alice_turn:
                        if vowel_count % 2 != 0:
                            valid_substrings.append(substring)
                    else:
                        if vowel_count % 2 == 0:
                            valid_substrings.append(substring)
            return valid_substrings

        def solve_game(current_s, is_alice_turn):
            if not current_s:
                return False # No moves possible, current player loses, so return False (not win)
            if (current_s, is_alice_turn) in memo:
                return memo[(current_s, is_alice_turn)]
            
            valid_moves = get_valid_moves(current_s, is_alice_turn)
            if not valid_moves:
                result = False # No valid move, current player loses
            else:
                win_possible = False
                for move in set(valid_moves): # Iterate over unique valid moves to avoid redundancy in state exploration
                    start_index = current_s.find(move)
                    while start_index != -1:
                        next_s = current_s[:start_index] + current_s[start_index+len(move):]
                        if not solve_game(next_s, not is_alice_turn):
                            win_possible = True
                            break
                        start_index = current_s.find(move, start_index + 1)
                    if win_possible:
                        break
                result = win_possible
                
            memo[(current_s, is_alice_turn)] = result
            return result

        return solve_game(s, True)