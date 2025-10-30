class Solution:
    def countWinningSequences(self, s: str) -> int:
        n = len(s)
        alice_moves_str = list(s)
        creatures = ['F', 'W', 'E']
        creature_to_index = {'F': 0, 'W': 1, 'E': 2}
        index_to_creature = {0: 'F', 1: 'W', 2: 'E'}
        alice_moves_index = [creature_to_index[move] for move in alice_moves_str]
        memo = {}
        mod = 10**9 + 7
        
        def get_score_diff_change(bob_move_index, alice_move_index):
            bob_move = index_to_creature[bob_move_index]
            alice_move = index_to_creature[alice_move_index]
            if (alice_move == 'E' and bob_move == 'F') or \
               (alice_move == 'F' and bob_move == 'W') or \
               (alice_move == 'W' and bob_move == 'E'):
                return 1
            if (alice_move == 'F' and bob_move == 'E') or \
               (alice_move == 'W' and bob_move == 'F') or \
               (alice_move == 'E' and bob_move == 'W'):
                return -1
            return 0
            
        def solve(round_num, last_bob_move_index, score_diff_index):
            if round_num == n:
                if score_diff_index > n:
                    return 1
                else:
                    return 0
            if (round_num, last_bob_move_index, score_diff_index) in memo:
                return memo[(round_num, last_bob_move_index, score_diff_index)]
            
            count = 0
            for current_bob_move_index in range(3):
                if last_bob_move_index == -1 or current_bob_move_index != last_bob_move_index:
                    alice_move_index = alice_moves_index[round_num]
                    score_change = get_score_diff_change(current_bob_move_index, alice_move_index)
                    next_score_diff_index = score_diff_index + score_change
                    if 0 <= next_score_diff_index <= 2 * n:
                        count = (count + solve(round_num + 1, current_bob_move_index, next_score_diff_index)) % mod
                        
            memo[(round_num, last_bob_move_index, score_diff_index)] = count
            return count
            
        initial_score_diff_index = n
        result = 0
        for first_move_index in range(3):
            result = (result + solve(1, first_move_index, initial_score_diff_index + get_score_diff_change(first_move_index, alice_moves_index[0]))) % mod

        return result