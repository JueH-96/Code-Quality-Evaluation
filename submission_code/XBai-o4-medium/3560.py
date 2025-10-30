from typing import List
from functools import lru_cache
from collections import deque

class Solution:
    def maxMoves(self, kx: int, ky: int, positions: List[List[int]]) -> int:
        m = len(positions)
        pawns = positions  # list of [x, y] for each pawn
        
        # Precompute minimal steps between positions using memoization
        @lru_cache(maxsize=None)
        def get_steps(x1, y1, x2, y2):
            directions = [ (-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1) ]
            visited = [ [ -1 ] * 50 for _ in range(50) ]
            q = deque()
            q.append( (x1, y1) )
            visited[x1][y1] = 0
            while q:
                cx, cy = q.popleft()
                if cx == x2 and cy == y2:
                    return visited[cx][cy]
                for dx, dy in directions:
                    nx = cx + dx
                    ny = cy + dy
                    if 0 <= nx < 50 and 0 <= ny < 50:
                        if visited[nx][ny] == -1:
                            visited[nx][ny] = visited[cx][cy] + 1
                            q.append( (nx, ny) )
            # According to problem statement, there's always a way to reach
            return -1
        
        # Now define the dp function
        @lru_cache(maxsize=None)
        def dp(mask, x, y):
            if mask == 0:
                return 0
            # count of pawns remaining in mask
            count_remaining = bin(mask).count('1')
            turns_made = m - count_remaining
            if turns_made % 2 == 0:  # Alice's turn, maximize
                max_total = 0
                for i in range(m):
                    if mask & (1 << i):
                        # pawn i is present
                        px, py = pawns[i]
                        steps = get_steps(x, y, px, py)
                        new_mask = mask & ~(1 << i)
                        new_x, new_y = px, py
                        sub = dp(new_mask, new_x, new_y)
                        total = steps + sub
                        if total > max_total:
                            max_total = total
                return max_total
            else:  # Bob's turn, minimize
                min_total = float('inf')
                for i in range(m):
                    if mask & (1 << i):
                        px, py = pawns[i]
                        steps = get_steps(x, y, px, py)
                        new_mask = mask & ~(1 << i)
                        new_x, new_y = px, py
                        sub = dp(new_mask, new_x, new_y)
                        total = steps + sub
                        if total < min_total:
                            min_total = total
                return min_total
        
        initial_mask = (1 << m) - 1
        return dp(initial_mask, kx, ky)