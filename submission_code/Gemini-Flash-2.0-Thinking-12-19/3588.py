class Solution:
    def countWinningSequences(self, s: str) -> int:
        n = len(s)
        alice_moves = []
        for char in s:
            if char == 'F':
                alice_moves.append(0) # Fire Dragon
            elif char == 'W':
                alice_moves.append(1) # Water Serpent
            elif char == 'E':
                alice_moves.append(2) # Earth Golem
        
        points_diff_matrix = [
            [0, 1, -1], # Alice: Fire, Bob: Fire, Water, Earth
            [-1, 0, 1], # Alice: Water, Bob: Fire, Water, Earth
            [1, -1, 0]  # Alice: Earth, Bob: Fire, Water, Earth
        ]
        
        memo = {}
        mod = 10**9 + 7
        
        def get_winning_sequences(round_index, last_bob_move_index, current_score_diff):
            if round_index == n:
                return 1 if current_score_diff > 0 else 0
            if (round_index, last_bob_move_index, current_score_diff) in memo:
                return memo[(round_index, last_bob_move_index, current_score_diff)]
            
            count = 0
            alice_move_index = alice_moves[round_index]
            for bob_move_index in range(3):
                if last_bob_move_index != -1 and bob_move_index == last_bob_move_index:
                    continue
                score_change = points_diff_matrix[alice_move_index][bob_move_index]
                next_score_diff = current_score_diff + score_change
                count = (count + get_winning_sequences(round_index + 1, bob_move_index, next_score_diff)) % mod
                
            memo[(round_index, last_bob_move_index, current_score_diff)] = count
            return count
            
        return get_winning_sequences(0, -1, 0)