from collections import deque
class Solution:
    def maxMoves(self, kx: int, ky: int, positions: List[List[int]]) -> int:
        def knight_moves(x, y, target_x, target_y):
            q = deque([(x, y, 0)])
            visited = set()
            visited.add((x, y))
            
            while q:
                curr_x, curr_y, moves = q.popleft()
                
                if curr_x == target_x and curr_y == target_y:
                    return moves
                
                possible_moves = [
                    (curr_x + 2, curr_y + 1),
                    (curr_x + 2, curr_y - 1),
                    (curr_x - 2, curr_y + 1),
                    (curr_x - 2, curr_y - 1),
                    (curr_x + 1, curr_y + 2),
                    (curr_x + 1, curr_y - 2),
                    (curr_x - 1, curr_y + 2),
                    (curr_x - 1, curr_y - 2)
                ]
                
                for next_x, next_y in possible_moves:
                    if 0 <= next_x < 50 and 0 <= next_y < 50 and (next_x, next_y) not in visited:
                        q.append((next_x, next_y, moves + 1))
                        visited.add((next_x, next_y))
            return float('inf')

        def solve(current_x, current_y, pawns_left, memo):
            if not pawns_left:
                return 0
            
            state = (current_x, current_y, tuple(sorted(pawns_left)))
            if state in memo:
                return memo[state]
            
            max_moves = float('-inf')
            
            for i in range(len(pawns_left)):
                next_pawns_left = pawns_left[:i] + pawns_left[i+1:]
                
                pawn_x, pawn_y = pawns_left[i]
                moves = knight_moves(current_x, current_y, pawn_x, pawn_y)
                
                max_moves = max(max_moves, moves + solve(pawn_x, pawn_y, next_pawns_left, memo))
                
            memo[state] = max_moves
            return max_moves

        pawns = []
        for x, y in positions:
            pawns.append((x, y))
        
        memo = {}
        return solve(kx, ky, pawns, memo)