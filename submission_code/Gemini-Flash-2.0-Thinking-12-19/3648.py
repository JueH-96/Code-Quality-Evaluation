from typing import List

class Solution:
    def maxCollectedFruits(self, fruits: List[List[int]]) -> int:
        n = len(fruits)
        memo = {}

        def get_moves1(p):
            r, c = p
            moves = [(1, 1), (1, 0), (0, 1)]
            next_positions = []
            for dr, dc in moves:
                nr, nc = r + dr, c + dc
                if 0 <= nr < n and 0 <= nc < n:
                    next_positions.append((nr, nc))
            return next_positions

        def get_moves2(p):
            r, c = p
            moves = [(1, -1), (1, 0), (1, 1)]
            next_positions = []
            for dr, dc in moves:
                nr, nc = r + dr, c + dc
                if 0 <= nr < n and 0 <= nc < n:
                    next_positions.append((nr, nc))
            return next_positions

        def get_moves3(p):
            r, c = p
            moves = [(-1, 1), (0, 1), (1, 1)]
            next_positions = []
            for dr, dc in moves:
                nr, nc = r + dr, c + dc
                if 0 <= nr < n and 0 <= nc < n:
                    next_positions.append((nr, nc))
            return next_positions

        def solve(step, p1, p2, p3):
            if step == n - 1:
                return 0
            if (step, p1, p2, p3) in memo:
                return memo[(step, p1, p2, p3)]
            
            max_fruits = 0
            next_moves_p1 = get_moves1(p1)
            next_moves_p2 = get_moves2(p2)
            next_moves_p3 = get_moves3(p3)

            for next_p1 in next_moves_p1:
                for next_p2 in next_moves_p2:
                    for next_p3 in next_moves_p3:
                        current_positions = [next_p1, next_p2, next_p3]
                        current_fruits_val = 0
                        visited_pos = set()
                        for pos in current_positions:
                            if pos not in visited_pos:
                                current_fruits_val += fruits[pos[0]][pos[1]]
                                visited_pos.add(pos)
                        recursive_result = solve(step + 1, next_p1, next_p2, next_p3)
                        max_fruits = max(max_fruits, current_fruits_val + recursive_result)

            memo[(step, p1, p2, p3)] = max_fruits
            return max_fruits

        initial_positions = [(0, 0), (0, n - 1), (n - 1, 0)]
        initial_fruits = 0
        visited_initial = set()
        for r, c in initial_positions:
            if (r, c) not in visited_initial:
                initial_fruits += fruits[r][c]
                visited_initial.add((r, c))

        result = solve(0, initial_positions[0], initial_positions[1], initial_positions[2])
        return initial_fruits + result