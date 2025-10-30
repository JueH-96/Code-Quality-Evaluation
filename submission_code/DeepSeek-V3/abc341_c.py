# YOUR CODE HERE
import sys
from collections import deque

def main():
    H, W, N = map(int, sys.stdin.readline().split())
    T = sys.stdin.readline().strip()
    grid = [sys.stdin.readline().strip() for _ in range(H)]
    
    # Directions for L, R, U, D
    dir_map = {'L': (0, -1), 'R': (0, 1), 'U': (-1, 0), 'D': (1, 0)}
    
    # Reverse the moves to find possible starting positions
    # We need to find all positions (i,j) such that when we apply the reverse of T, we stay on land
    # So, we start from all possible positions and apply the reverse moves
    
    # Reverse the sequence of moves
    reversed_T = T[::-1]
    
    # Initialize the possible positions as all land cells
    possible = [[True for _ in range(W)] for _ in range(H)]
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '#':
                possible[i][j] = False
    
    # Apply the reversed moves
    for move in reversed_T:
        new_possible = [[False for _ in range(W)] for _ in range(H)]
        dx, dy = dir_map[move]
        for i in range(H):
            for j in range(W):
                if possible[i][j]:
                    ni = i + dx
                    nj = j + dy
                    if 0 <= ni < H and 0 <= nj < W and grid[ni][nj] == '.':
                        new_possible[ni][nj] = True
        possible = new_possible
    
    # Count the number of True in possible
    count = 0
    for i in range(H):
        for j in range(W):
            if possible[i][j]:
                count += 1
    print(count)

if __name__ == "__main__":
    main()