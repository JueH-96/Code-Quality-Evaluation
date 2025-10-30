class Solution:
    def countWinningSequences(self, s: str) -> int:
        MOD = 10**9 + 7
        n = len(s)
        # Mapping moves to indices
        move_mapping = {'F': 0, 'W': 1, 'E': 2}
        moves = ['F', 'W', 'E']
        # Define winning pairs where the first move beats the second
        win_pairs = {('F', 'E'), ('W', 'F'), ('E', 'W')}
        # Initialize DP arrays
        # dp_prev[last_move][diff_shifted]
        dp_prev = [ [0] * 2001 for _ in range(4) ]  # 0:F,1:W,2:E,3:None
        dp_prev[3][1000] = 1  # Starting with no last move and diff=0
        dp_next = [ [0] * 2001 for _ in range(4) ]
        
        for i in range(n):
            a_i = s[i]
            for last_move_index in range(4):
                for diff_shifted in range(2001):
                    val = dp_prev[last_move_index][diff_shifted]
                    if val == 0:
                        continue
                    for move_idx, move in enumerate(moves):
                        if last_move_index < 3 and move == moves[last_move_index]:
                            continue  # Cannot repeat the same move
                        # Determine the outcome
                        if (move, a_i) in win_pairs:
                            delta = 1
                        elif (a_i, move) in win_pairs:
                            delta = -1
                        else:
                            delta = 0
                        new_diff_shifted = diff_shifted + delta
                        if 0 <= new_diff_shifted <= 2000:
                            dp_next[move_idx][new_diff_shifted] = (dp_next[move_idx][new_diff_shifted] + val) % MOD
            # Reset dp_prev and dp_next for the next iteration
            for lm in range(4):
                for d in range(2001):
                    dp_prev[lm][d] = dp_next[lm][d]
                    dp_next[lm][d] = 0
        # Calculate the total number of sequences where diff > 0
        result = 0
        for last_move_index in range(3):  # Only 'F', 'W', 'E' as last moves
            for diff_shifted in range(1001, 2001):
                result = (result + dp_prev[last_move_index][diff_shifted]) % MOD
        return result