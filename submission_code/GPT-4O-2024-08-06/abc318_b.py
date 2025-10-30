# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    rectangles = []
    index = 1
    for _ in range(N):
        A = int(data[index])
        B = int(data[index + 1])
        C = int(data[index + 2])
        D = int(data[index + 3])
        rectangles.append((A, B, C, D))
        index += 4
    
    # Create a grid to mark covered areas
    grid = [[False] * 101 for _ in range(101)]
    
    # Mark the grid for each rectangle
    for A, B, C, D in rectangles:
        for x in range(A, B):
            for y in range(C, D):
                grid[x][y] = True
    
    # Calculate the total covered area
    covered_area = 0
    for x in range(101):
        for y in range(101):
            if grid[x][y]:
                covered_area += 1
    
    # Output the result
    print(covered_area)

main()