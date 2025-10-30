class Solution:
    def countWinningSequences(self, s: str) -> int:
        n = len(s)
        mod = 10**9 + 7
        
        def get_points(alice_move, bob_move):
            if alice_move == bob_move:
                return 0
            elif (alice_move == 'F' and bob_move == 'E') or \
                 (alice_move == 'W' and bob_move == 'F') or \
                 (alice_move == 'E' and bob_move == 'W'):
                return 1
            else:
                return -1
            
        
        @lru_cache(None)
        def count_winning(index, prev_move, bob_points):
            if index == n:
                return 1 if bob_points > 0 else 0
            
            ans = 0
            for move in ['F', 'W', 'E']:
                if move != prev_move:
                    points = get_points(s[index], move)
                    ans = (ans + count_winning(index + 1, move, bob_points + points)) % mod
            return ans
        
        return count_winning(0, None, 0)