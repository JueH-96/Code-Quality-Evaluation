# YOUR CODE HERE
def is_valid(x, y, H, W):
    return 1 <= x <= H and 1 <= y <= W

def can_reach(start_x, start_y, H, W, N, T, grid):
    x, y = start_x, start_y
    for move in T:
        if move == 'L':
            y -= 1
        elif move == 'R':
            y += 1
        elif move == 'U':
            x -= 1
        elif move == 'D':
            x += 1
        if not is_valid(x, y, H, W) or grid[x-1][y-1] == '#':
            return False
    return True

def main():
    import sys
    input = sys.stdin.read().split()
    H = int(input[0])
    W = int(input[1])
    N = int(input[2])
    T = input[3]
    grid = input[4:H+4]
    
    possible_positions = 0
    
    for i in range(1, H+1):
        for j in range(1, W+1):
            if grid[i-1][j-1] == '.':
                if can_reach(i, j, H, W, N, T, grid):
                    possible_positions += 1
    
    print(possible_positions)

main()