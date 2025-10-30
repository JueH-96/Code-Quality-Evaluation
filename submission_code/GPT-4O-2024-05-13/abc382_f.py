# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    H = int(data[0])
    W = int(data[1])
    N = int(data[2])
    
    bars = []
    index = 3
    for i in range(N):
        R = int(data[index])
        C = int(data[index + 1])
        L = int(data[index + 2])
        bars.append((R, C, L))
        index += 3
    
    # Create a grid to track the final positions of the bars
    grid = [[0] * W for _ in range(H)]
    
    # Place the bars initially on the grid
    for i in range(N):
        R, C, L = bars[i]
        for j in range(L):
            grid[R - 1][C - 1 + j] = i + 1
    
    # Simulate the falling process
    for t in range(H):
        for i in range(N):
            R, C, L = bars[i]
            if R < H:
                can_fall = True
                for j in range(L):
                    if grid[R][C - 1 + j] != 0:
                        can_fall = False
                        break
                if can_fall:
                    for j in range(L):
                        grid[R - 1][C - 1 + j] = 0
                    for j in range(L):
                        grid[R][C - 1 + j] = i + 1
                    bars[i] = (R + 1, C, L)
    
    # Output the final row positions of each bar
    for i in range(N):
        print(bars[i][0])

if __name__ == "__main__":
    main()