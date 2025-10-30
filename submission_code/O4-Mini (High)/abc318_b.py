import sys

def main():
    input = sys.stdin.readline
    N = int(input())
    # Create a 100x100 grid of booleans to mark covered unit squares
    grid = [[False] * 100 for _ in range(100)]
    
    for _ in range(N):
        A, B, C, D = map(int, input().split())
        # Mark all unit squares [x, x+1) x [y, y+1) covered by this rectangle
        for x in range(A, B):
            for y in range(C, D):
                grid[x][y] = True
    
    # Count the number of covered squares
    area = 0
    for x in range(100):
        for y in range(100):
            if grid[x][y]:
                area += 1
    
    print(area)

# Call main to execute
main()