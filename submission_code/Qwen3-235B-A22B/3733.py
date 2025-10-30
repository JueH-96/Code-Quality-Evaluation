from typing import List

class Solution:
    def lenOfVDiagonal(self, grid: List[List[int]]) -> int:
        n = len(grid)
        if n == 0:
            return 0
        m = len(grid[0])
        max_len = 0

        # Define the four diagonal directions: down-right, down-left, up-right, up-left
        directions = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
        # Mapping of clockwise turns: each direction turns to the next in clockwise order
        clockwise_dir = [1, 2, 3, 0]

        def count_steps(x: int, y: int, dx: int, dy: int, start_step: int) -> int:
            steps = 0
            current_x, current_y = x, y
            current_s = start_step
            while 0 <= current_x < n and 0 <= current_y < m:
                # Determine expected value based on current step
                if current_s == 0:
                    expected = 1
                else:
                    expected = 2 if (current_s % 2 == 1) else 0
                if grid[current_x][current_y] != expected:
                    break
                steps += 1
                # Move to next cell in direction and increment step
                current_x += dx
                current_y += dy
                current_s += 1
            return steps

        for i in range(n):
            for j in range(m):
                if grid[i][j] == 1:
                    for d in range(4):
                        dx_dir, dy_dir = directions[d]
                        # Calculate the maximum length of a straight line in direction d
                        straight_len = count_steps(i, j, dx_dir, dy_dir, 0)
                        if straight_len > max_len:
                            max_len = straight_len
                        # Consider all possible turns after k steps in direction d
                        if straight_len >= 1:
                            for k in range(1, straight_len + 1):
                                turn_x = i + dx_dir * (k - 1)
                                turn_y = j + dy_dir * (k - 1)
                                if turn_x < 0 or turn_x >= n or turn_y < 0 or turn_y >= m:
                                    continue
                                # Determine the new direction after clockwise turn
                                new_d = clockwise_dir[d]
                                new_dx, new_dy = directions[new_d]
                                # Calculate steps in the new direction starting from step = k
                                steps_new = count_steps(turn_x, turn_y, new_dx, new_dy, k)
                                total_len = k + steps_new
                                if total_len > max_len:
                                    max_len = total_len

        return max_len