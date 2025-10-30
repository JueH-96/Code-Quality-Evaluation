import sys
sys.setrecursionlimit(1000000)

def main():
    grid = []
    for _ in range(3):
        grid.append(list(map(int, sys.stdin.readline().split())))

    # Directions for checking three in a row
    directions = [
        [(0,0),(0,1),(0,2)],  # Rows
        [(1,0),(1,1),(1,2)],
        [(2,0),(2,1),(2,2)],
        [(0,0),(1,0),(2,0)],  # Columns
        [(0,1),(1,1),(2,1)],
        [(0,2),(1,2),(2,2)],
        [(0,0),(1,1),(2,2)],  # Diagonals
        [(0,2),(1,1),(2,0)]
    ]

    from functools import lru_cache

    # Represent the board as a tuple of tuples
    initial_board = tuple(tuple(row) for row in grid)
    # Use a mask to represent painted cells
    # mask: 0babcdefghi, where each bit represents a cell (0 to 8)
    # player: 0 for Takahashi, 1 for Aoki
    # scores: tuple(takahashi_score, aoki_score)

    @lru_cache(maxsize=None)
    def minimax(mask, player, takahashi_score, aoki_score):
        # Check if any player has won
        # Check for Takahashi's win
        for line in directions:
            cells = [mask >> (i) & 1 for i in [pos[0]*3 + pos[1] for pos in line]]
            if all(cell == 1 for cell in cells):
                return 1  # Takahashi wins
        # Check for Aoki's win
        for line in directions:
            cells = [mask >> (i+9) & 1 for i in [pos[0]*3 + pos[1] for pos in line]]
            if all(cell == 1 for cell in cells):
                return -1  # Aoki wins
        # Check if all cells are painted
        if mask >= (1 << 9):
            if takahashi_score > aoki_score:
                return 1  # Takahashi wins by score
            else:
                return -1  # Aoki wins by score
        # Determine whose turn it is
        current_player = player
        # Takahashi's turn
        if current_player == 0:
            best = -2  # Less than -1
            for i in range(9):
                if not (mask & (1 << i)):
                    # Paint the cell for Takahashi
                    new_mask = mask | (1 << i)
                    # Calculate new score
                    new_takahashi_score = takahashi_score + grid[i//3][i%3]
                    # Recurse
                    result = minimax(new_mask, 1, new_takahashi_score, aoki_score)
                    best = max(best, result)
                    if best == 1:
                        break  # Prune
            return best
        else:
            best = 2  # More than 1
            for i in range(9):
                if not (mask >> (i+9) & 1):
                    # Paint the cell for Aoki
                    new_mask = mask | (1 << (i+9))
                    # Calculate new score
                    new_aoki_score = aoki_score + grid[i//3][i%3]
                    # Recurse
                    result = minimax(new_mask, 0, takahashi_score, new_aoki_score)
                    best = min(best, result)
                    if best == -1:
                        break  # Prune
            return best

    result = minimax(0, 0, 0, 0)
    if result == 1:
        print("Takahashi")
    else:
        print("Aoki")

if __name__ == "__main__":
    main()