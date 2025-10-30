from collections import deque
from typing import List

# Precompute the steps once
steps = [[None for _ in range(50)] for __ in range(50)]

def precompute_steps():
    global steps
    for kx in range(50):
        for ky in range(50):
            steps_table = [[-1 for _ in range(50)] for __ in range(50)]
            queue = deque()
            queue.append((kx, ky))
            steps_table[kx][ky] = 0
            while queue:
                x, y = queue.popleft()
                current_steps = steps_table[x][y]
                # Generate all possible knight moves
                for dx, dy in [(-2, -1), (-2, 1), (-1, -2), (-1, 2),
                               (1, -2), (1, 2), (2, -1), (2, 1)]:
                    nx = x + dx
                    ny = y + dy
                    if 0 <= nx < 50 and 0 <= ny < 50:
                        if steps_table[nx][ny] == -1:
                            steps_table[nx][ny] = current_steps + 1
                            queue.append((nx, ny))
            steps[kx][ky] = steps_table

precompute_steps()

class Solution:
    def maxMoves(self, kx: int, ky: int, positions: List[List[int]]) -> int:
        n = len(positions)
        if n == 0:
            return 0
        
        memo = {}
        
        def dfs(current_kx, current_ky, current_mask, is_alice_turn):
            if current_mask == 0:
                return 0
            key = (current_kx, current_ky, current_mask, is_alice_turn)
            if key in memo:
                return memo[key]
            
            if is_alice_turn:
                max_total = 0
                for i in range(n):
                    if not (current_mask & (1 << i)):
                        continue
                    x = positions[i][0]
                    y = positions[i][1]
                    minimal_steps = steps[current_kx][current_ky][x][y]
                    new_mask = current_mask ^ (1 << i)
                    new_kx, new_ky = x, y
                    total = minimal_steps + dfs(new_kx, new_ky, new_mask, False)
                    if total > max_total:
                        max_total = total
                memo[key] = max_total
                return max_total
            else:
                min_total = float('inf')
                for i in range(n):
                    if not (current_mask & (1 << i)):
                        continue
                    x = positions[i][0]
                    y = positions[i][1]
                    minimal_steps = steps[current_kx][current_ky][x][y]
                    new_mask = current_mask ^ (1 << i)
                    new_kx, new_ky = x, y
                    total = minimal_steps + dfs(new_kx, new_ky, new_mask, True)
                    if total < min_total:
                        min_total = total
                if min_total == float('inf'):
                    min_total = 0
                memo[key] = min_total
                return min_total
        
        initial_mask = (1 << n) - 1
        result = dfs(kx, ky, initial_mask, True)
        return result