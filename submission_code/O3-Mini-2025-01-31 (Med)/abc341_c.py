import sys
import numpy as np

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    H = int(data[0])
    W = int(data[1])
    N = int(data[2])
    T = data[3]
    grid_lines = data[4:]
    
    # Build a boolean grid representing land (True) and sea (False).
    # A cell is land if it is '.', sea if it is '#'.
    land = np.array([[ch == '.' for ch in line] for line in grid_lines], dtype=bool)
    
    # Initially, any cell that is land is a possible crash-landing cell.
    # dp will represent the set of cells where Takahashi can be after some moves.
    dp = land.copy()
    
    # Process each move in the instruction string.
    # We update dp via shifting. For example, if the move is 'L',
    # then if previously dp[r][c+1] was possible, then after moving left, dp[r][c] is possible.
    for move in T:
        new_dp = np.zeros_like(dp)
        if move == 'L':
            new_dp[:, :W-1] = dp[:, 1:]
        elif move == 'R':
            new_dp[:, 1:] = dp[:, :W-1]
        elif move == 'U':
            new_dp[:H-1, :] = dp[1:, :]
        elif move == 'D':
            new_dp[1:, :] = dp[:H-1, :]
        # The new positions must be land.
        dp = new_dp & land

    # dp now contains True for all cells that could be Takahashi's final position.
    print(int(np.count_nonzero(dp)))

if __name__ == '__main__':
    main()