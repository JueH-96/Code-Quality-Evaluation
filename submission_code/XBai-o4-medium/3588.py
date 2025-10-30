class Solution:
    def countWinningSequences(self, s: str) -> int:
        MOD = 10**9 + 7
        n = len(s)
        if n == 0:
            return 0
        shift = n
        
        # Precompute the a_moves
        a_moves = []
        for c in s:
            if c == 'F':
                a_moves.append(0)
            elif c == 'W':
                a_moves.append(1)
            else:
                a_moves.append(2)
        
        # Initialize DP table for the first round
        prev_dp = [[0] * (2 * shift + 1) for _ in range(3)]
        a = a_moves[0]
        for b in range(3):
            if a == b:
                delta = 0
            else:
                if (b == 0 and a == 2) or (b == 1 and a == 0) or (b == 2 and a == 1):
                    delta = 1
                else:
                    delta = -1
            shifted = delta + shift
            prev_dp[b][shifted] = 1
        
        # Process remaining rounds
        for i in range(1, n):
            curr_dp = [[0] * (2 * shift + 1) for _ in range(3)]
            a_curr = a_moves[i]
            for prev_move in range(3):
                for delta_shift in range(2 * shift + 1):
                    count = prev_dp[prev_move][delta_shift]
                    if count == 0:
                        continue
                    current_delta = delta_shift - shift
                    # Try all possible current_move != prev_move
                    for current_move in range(3):
                        if current_move == prev_move:
                            continue
                        # Compute delta_round
                        if a_curr == current_move:
                            delta_round = 0
                        else:
                            if (current_move == 0 and a_curr == 2) or \
                               (current_move == 1 and a_curr == 0) or \
                               (current_move == 2 and a_curr == 1):
                                delta_round = 1
                            else:
                                delta_round = -1
                        new_delta = current_delta + delta_round
                        new_shift = new_delta + shift
                        # Update curr_dp
                        curr_dp[current_move][new_shift] = (curr_dp[current_move][new_shift] + count) % MOD
            prev_dp = curr_dp
        
        # Calculate total number of winning sequences
        total = 0
        for move in range(3):
            for delta_shift in range(2 * shift + 1):
                delta = delta_shift - shift
                if delta > 0:
                    total = (total + prev_dp[move][delta_shift]) % MOD
        return total