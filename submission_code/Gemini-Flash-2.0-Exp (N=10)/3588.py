class Solution:
    def countWinningSequences(self, s: str) -> int:
        n = len(s)
        MOD = 10**9 + 7
        
        def get_winner(alice_move, bob_move):
            if alice_move == bob_move:
                return 0
            if (alice_move == 'F' and bob_move == 'E') or \
               (alice_move == 'W' and bob_move == 'F') or \
               (alice_move == 'E' and bob_move == 'W'):
                return -1
            return 1

        def count_wins(index, bob_seq, alice_score, bob_score):
            if index == n:
                return 1 if bob_score > alice_score else 0
            
            count = 0
            for bob_move in ['F', 'W', 'E']:
                if not bob_seq or bob_seq[-1] != bob_move:
                    winner = get_winner(s[index], bob_move)
                    new_alice_score = alice_score
                    new_bob_score = bob_score
                    if winner == 1:
                        new_bob_score += 1
                    elif winner == -1:
                        new_alice_score += 1
                    count = (count + count_wins(index + 1, bob_seq + bob_move, new_alice_score, new_bob_score)) % MOD
            return count
        
        return count_wins(0, "", 0, 0)