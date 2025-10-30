# YOUR CODE HERE
H, W = map(int, input().split())
Si, Sj = map(int, input().split())
grid = [input() for _ in range(H)]
X = input()

# Convert Si, Sj to 0-based indexing
Si -= 1
Sj -= 1

for move in X:
    if move == 'L' and Sj > 0 and grid[Si][Sj-1] == '.':
        Sj -= 1
    elif move == 'R' and Sj < W-1 and grid[Si][Sj+1] == '.':
        Sj += 1
    elif move == 'U' and Si > 0 and grid[Si-1][Sj] == '.':
        Si -= 1
    elif move == 'D' and Si < H-1 and grid[Si+1][Sj] == '.':
        Si += 1

# Convert back to 1-based indexing for output
print(Si+1, Sj+1)