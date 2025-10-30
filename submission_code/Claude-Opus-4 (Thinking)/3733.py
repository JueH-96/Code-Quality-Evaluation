class Solution:
    def lenOfVDiagonal(self, grid: List[List[int]]) -> int:
        n, m = len(grid), len(grid[0])
        
        # Diagonal directions: down-right, down-left, up-left, up-right
        directions = [(1, 1), (1, -1), (-1, -1), (-1, 1)]
        
        # Clockwise turn mapping
        clockwise_turn = {
            (1, 1): (1, -1),    # down-right to down-left
            (1, -1): (-1, -1),  # down-left to up-left
            (-1, -1): (-1, 1),  # up-left to up-right
            (-1, 1): (1, 1)     # up-right to down-right
        }
        
        # Expected sequence: 1, 2, 0, 2, 0, 2, 0, ...
        def get_expected(index):
            if index == 0:
                return 1
            elif index == 1:
                return 2
            else:
                return 0 if index % 2 == 0 else 2
        
        max_length = 0
        
        # Try starting from each cell with value 1
        for i in range(n):
            for j in range(m):
                if grid[i][j] == 1:
                    # Try each initial direction
                    for di, dj in directions:
                        # For each starting direction, we'll try:
                        # 1. Going straight (no turn)
                        # 2. Turning at each possible position
                        
                        # First, calculate the straight path
                        straight_length = 1
                        ni, nj = i + di, j + dj
                        seq_idx = 1
                        positions = [(i, j)]  # Store positions for potential turns
                        
                        while 0 <= ni < n and 0 <= nj < m and grid[ni][nj] == get_expected(seq_idx):
                            straight_length += 1
                            positions.append((ni, nj))
                            seq_idx += 1
                            ni += di
                            nj += dj
                        
                        max_length = max(max_length, straight_length)
                        
                        # Now try turning at each position
                        new_di, new_dj = clockwise_turn[(di, dj)]
                        
                        for turn_idx in range(len(positions)):
                            # Length up to turn position
                            length = turn_idx + 1
                            
                            # Continue from turn position in new direction
                            ti, tj = positions[turn_idx]
                            ni, nj = ti + new_di, tj + new_dj
                            seq_idx = turn_idx + 1
                            
                            while 0 <= ni < n and 0 <= nj < m and grid[ni][nj] == get_expected(seq_idx):
                                length += 1
                                seq_idx += 1
                                ni += new_di
                                nj += new_dj
                            
                            max_length = max(max_length, length)
        
        return max_length