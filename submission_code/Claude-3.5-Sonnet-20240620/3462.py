class Solution:
    def doesAliceWin(self, s: str) -> bool:
        vowels = set('aeiou')
        
        def count_vowels(substring):
            return sum(1 for char in substring if char in vowels)
        
        def can_make_move(player, string):
            target = 1 if player == 'Alice' else 0
            for i in range(len(string)):
                for j in range(i + 1, len(string) + 1):
                    if count_vowels(string[i:j]) % 2 == target:
                        return True
            return False
        
        def play_game(player, string):
            if not can_make_move(player, string):
                return player == 'Bob'
            
            next_player = 'Bob' if player == 'Alice' else 'Alice'
            
            for i in range(len(string)):
                for j in range(i + 1, len(string) + 1):
                    substring = string[i:j]
                    vowel_count = count_vowels(substring)
                    if (player == 'Alice' and vowel_count % 2 == 1) or (player == 'Bob' and vowel_count % 2 == 0):
                        new_string = string[:i] + string[j:]
                        if not play_game(next_player, new_string):
                            return True
            
            return False
        
        return play_game('Alice', s)