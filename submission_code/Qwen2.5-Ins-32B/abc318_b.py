import sys

def read_ints(): return map(int, sys.stdin.readline().strip().split())

def calculate_covered_area(N, rectangles):
    grid = [[0 for _ in range(101)] for _ in range(101)]
    
    for A, B, C, D in rectangles:
        for x in range(A, B):
            for y in range(C, D):
                grid[x][y] = 1
    
    return sum(sum(row) for row in grid)

N = int(input())
rectangles = [read_ints() for _ in range(N)]
print(calculate_covered_area(N, rectangles))