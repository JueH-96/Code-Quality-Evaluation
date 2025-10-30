# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    sheets = []
    index = 1
    for _ in range(N):
        A = int(data[index])
        B = int(data[index+1])
        C = int(data[index+2])
        D = int(data[index+3])
        sheets.append((A, B, C, D))
        index += 4
    
    # Initialize a grid to represent the plane
    # Since 0 <= x, y <= 100, we can use a 101x101 grid
    grid = [[0] * 101 for _ in range(101)]
    
    for sheet in sheets:
        A, B, C, D = sheet
        for x in range(A, B):
            for y in range(C, D):
                grid[x][y] = 1
    
    # Calculate the total area
    S = 0
    for x in range(101):
        for y in range(101):
            if grid[x][y] == 1:
                S += 1
    
    print(S)

if __name__ == "__main__":
    main()