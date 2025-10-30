class Solution:
    def maxMoves(self, kx: int, ky: int, positions: List[List[int]]) -> int:
        
        def bfs(start_x, start_y, end_x, end_y):
            queue = [(start_x, start_y, 0)]
            visited = set()
            visited.add((start_x, start_y))
            while queue:
                x, y, dist = queue.pop(0)
                if x == end_x and y == end_y:
                    return dist
                moves = [(x + 1, y + 2), (x + 1, y - 2), (x - 1, y + 2), (x - 1, y - 2),
                         (x + 2, y + 1), (x + 2, y - 1), (x - 2, y + 1), (x - 2, y - 1)]
                for next_x, next_y in moves:
                    if 0 <= next_x < 50 and 0 <= next_y < 50 and (next_x, next_y) not in visited:
                        visited.add((next_x, next_y))
                        queue.append((next_x, next_y, dist + 1))
            return -1

        n = len(positions)
        
        def minimax(current_player, current_kx, current_ky, remaining_positions):
            if not remaining_positions:
                return 0

            if current_player == 0:  # Alice's turn (maximizing)
                max_moves = -1
                for i in range(len(remaining_positions)):
                    next_pos = remaining_positions[i]
                    moves = bfs(current_kx, current_ky, next_pos[0], next_pos[1])
                    next_remaining_positions = remaining_positions[:i] + remaining_positions[i+1:]
                    max_moves = max(max_moves, moves + minimax(1, next_pos[0], next_pos[1], next_remaining_positions))
                return max_moves
            else:  # Bob's turn (minimizing)
                min_moves = float('inf')
                for i in range(len(remaining_positions)):
                    next_pos = remaining_positions[i]
                    moves = bfs(current_kx, current_ky, next_pos[0], next_pos[1])
                    next_remaining_positions = remaining_positions[:i] + remaining_positions[i+1:]
                    min_moves = min(min_moves, moves + minimax(0, next_pos[0], next_pos[1], next_remaining_positions))
                return min_moves

        return minimax(0, kx, ky, positions)