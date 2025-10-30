class Solution:
    def maxMoves(self, kx: int, ky: int, positions: List[List[int]]) -> int:
        def min_moves(kx, ky, px, py):
            moves = [
                (2, 1), (2, -1), (-2, 1), (-2, -1),
                (1, 2), (1, -2), (-1, 2), (-1, -2)
            ]
            queue = [(0, kx, ky)]
            visited = set()
            while queue:
                dist, x, y = queue.pop(0)
                if (x, y) == (px, py):
                    return dist
                for dx, dy in moves:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < 50 and 0 <= ny < 50 and (nx, ny) not in visited:
                        visited.add((nx, ny))
                        queue.append((dist + 1, nx, ny))
            return float('inf')

        def dfs(pawns, turn):
            if not pawns:
                return 0
            if turn == 'Alice':
                return max(min_moves(kx, ky, px, py) + dfs(pawns - {pos}, 'Bob') for px, py in pawns)
            else:
                return min(min_moves(kx, ky, px, py) + dfs(pawns - {pos}, 'Alice') for px, py in pawns)

        return dfs(set(map(tuple, positions)), 'Alice')