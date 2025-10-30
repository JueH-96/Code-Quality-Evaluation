class Solution:
    def countWinningSequences(self, s: str) -> int:
        mod = 10**9 + 7
        n = len(s)
        ans = 0
        
        def calculate_score(alice_moves, bob_moves):
            alice_score = 0
            bob_score = 0
            for i in range(len(alice_moves)):
                if alice_moves[i] == 'F' and bob_moves[i] == 'E':
                    bob_score += 1
                elif alice_moves[i] == 'W' and bob_moves[i] == 'F':
                    bob_score += 1
                elif alice_moves[i] == 'E' and bob_moves[i] == 'W':
                    bob_score += 1
                elif bob_moves[i] == 'F' and alice_moves[i] == 'E':
                    alice_score += 1
                elif bob_moves[i] == 'W' and alice_moves[i] == 'F':
                    alice_score += 1
                elif bob_moves[i] == 'E' and alice_moves[i] == 'W':
                    alice_score += 1
            return alice_score, bob_score

        def generate_bob_moves(current_moves, index):
            nonlocal ans
            if index == n:
                alice_score, bob_score = calculate_score(s, "".join(current_moves))
                if bob_score > alice_score:
                    ans = (ans + 1) % mod
                return
            
            moves = ['F', 'W', 'E']
            if index > 0:
                last_move = current_moves[-1]
                moves.remove(last_move)

            for move in moves:
                generate_bob_moves(current_moves + [move], index + 1)

        generate_bob_moves([], 0)
        return ans