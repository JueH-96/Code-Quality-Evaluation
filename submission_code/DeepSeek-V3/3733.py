class Solution:
    def lenOfVDiagonal(self, grid: List[List[int]]) -> int:
        if not grid or not grid[0]:
            return 0
        rows = len(grid)
        cols = len(grid[0])
        max_len = 0
        
        # Directions: four diagonal directions (dr, dc)
        # 1: down-right (1, 1)
        # 2: down-left (1, -1)
        # 3: up-right (-1, 1)
        # 4: up-left (-1, -1)
        directions = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
        
        for i in range(rows):
            for j in range(cols):
                if grid[i][j] == 1:
                    for dr, dc in directions:
                        # Explore without turning
                        length = 1
                        ni, nj = i + dr, j + dc
                        step = 1  # next expected value: 2 if step is odd, 0 if even
                        while 0 <= ni < rows and 0 <= nj < cols:
                            expected = 2 if step % 2 == 1 else 0
                            if grid[ni][nj] == expected:
                                length += 1
                                step += 1
                                ni += dr
                                nj += dc
                            else:
                                break
                        if length > max_len:
                            max_len = length
                        
                        # Explore with a turn after each possible step
                        ni, nj = i, j
                        current_len = 1
                        step = 1
                        ni += dr
                        nj += dc
                        turn_possible = False
                        while 0 <= ni < rows and 0 <= nj < cols:
                            expected = 2 if step % 2 == 1 else 0
                            if grid[ni][nj] == expected:
                                current_len += 1
                                step += 1
                                # Try turning 90 degrees clockwise
                                # Original direction (dr, dc)
                                # Clockwise 90 degree turns:
                                # (dr, dc) -> (dc, -dr) or (-dc, dr) depending on original direction?
                                # For example, direction (1, 1) turns into (1, -1) or (-1, 1)?
                                # Need to find the correct new direction.
                                # For a clockwise turn:
                                # (dr, dc) becomes (dc, -dr)
                                new_dr = dc
                                new_dc = -dr
                                new_ni = ni + new_dr
                                new_nj = nj + new_dc
                                new_step = step
                                temp_len = current_len
                                temp_ni, temp_nj = new_ni, new_nj
                                while 0 <= temp_ni < rows and 0 <= temp_nj < cols:
                                    expected_new = 2 if new_step % 2 == 1 else 0
                                    if grid[temp_ni][temp_nj] == expected_new:
                                        temp_len += 1
                                        new_step += 1
                                        temp_ni += new_dr
                                        temp_nj += new_dc
                                    else:
                                        break
                                if temp_len > max_len:
                                    max_len = temp_len
                                ni += dr
                                nj += dc
                            else:
                                break
        return max_len