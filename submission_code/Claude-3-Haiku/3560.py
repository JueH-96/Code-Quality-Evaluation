from collections import deque

class Solution:
    def maxMoves(self, kx: int, ky: int, positions: List[List[int]]) -> int:
        # Define the eight possible knight moves
        dx = [-2, -1, 1, 2, 2, 1, -1, -2]
        dy = [1, 2, 2, 1, -1, -2, -2, -1]
        
        # Function to get the minimum number of moves to reach a pawn
        def bfs(x, y, tx, ty):
            queue = deque([(x, y, 0)])
            visited = set()
            while queue:
                cx, cy, moves = queue.popleft()
                if (cx, cy) == (tx, ty):
                    return moves
                if (cx, cy) in visited:
                    continue
                visited.add((cx, cy))
                for i in range(8):
                    nx, ny = cx + dx[i], cy + dy[i]
                    if 0 <= nx < 50 and 0 <= ny < 50:
                        queue.append((nx, ny, moves + 1))
            return float('inf')
        
        # Sort the pawns by the minimum number of moves required to reach them
        pawns = sorted(positions, key=lambda p: bfs(kx, ky, p[0], p[1]))
        
        # Play the game optimally
        total_moves = 0
        for x, y in pawns:
            total_moves += bfs(kx, ky, x, y)
            kx, ky = x, y
        
        return total_moves