class Solution:
    def countWinningSequences(self, s: str) -> int:
        MOD = 10**9 + 7
        n = len(s)
        
        # Map creatures to indices: F=0, W=1, E=2
        creature_to_idx = {'F': 0, 'W': 1, 'E': 2}
        idx_to_creature = ['F', 'W', 'E']
        
        # Function to get points for Bob given Alice's and Bob's moves
        def get_points(alice_move, bob_move):
            if alice_move == bob_move:
                return 0
            # Bob wins cases
            if (alice_move == 'F' and bob_move == 'W') or \
               (alice_move == 'W' and bob_move == 'E') or \
               (alice_move == 'E' and bob_move == 'F'):
                return 1
            # Alice wins
            return -1
        
        # Memoization cache
        memo = {}
        
        def dp(round_idx, last_bob_move, score_diff):
            if round_idx == n:
                # Bob wins if score_diff > 0
                return 1 if score_diff > 0 else 0
            
            if (round_idx, last_bob_move, score_diff) in memo:
                return memo[(round_idx, last_bob_move, score_diff)]
            
            alice_move = s[round_idx]
            result = 0
            
            # Try all possible moves for Bob
            for bob_move in ['F', 'W', 'E']:
                # Bob cannot repeat the same move consecutively
                if round_idx > 0 and bob_move == last_bob_move:
                    continue
                
                # Calculate points for this round
                points = get_points(alice_move, bob_move)
                new_score_diff = score_diff + points
                
                # Pruning: if score difference becomes too negative, Bob cannot win
                # Maximum possible points Bob can get in remaining rounds is (n - round_idx - 1)
                if new_score_diff + (n - round_idx - 1) <= 0:
                    continue
                
                result = (result + dp(round_idx + 1, bob_move, new_score_diff)) % MOD
            
            memo[(round_idx, last_bob_move, score_diff)] = result
            return result
        
        # Start with round 0, no previous move (use None), score difference 0
        return dp(0, None, 0)