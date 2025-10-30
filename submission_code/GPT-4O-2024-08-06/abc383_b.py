# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    H = int(data[0])
    W = int(data[1])
    D = int(data[2])
    grid = data[3:]
    
    floor_cells = []
    
    # Collect all floor cells
    for i in range(H):
        for j in range(W):
            if grid[i][j] == '.':
                floor_cells.append((i, j))
    
    max_humidified = 0
    
    # Iterate over all pairs of floor cells
    for idx1 in range(len(floor_cells)):
        for idx2 in range(idx1 + 1, len(floor_cells)):
            humidifier1 = floor_cells[idx1]
            humidifier2 = floor_cells[idx2]
            
            humidified_set = set()
            
            # Check cells humidified by the first humidifier
            for i in range(H):
                for j in range(W):
                    if grid[i][j] == '.':
                        if abs(i - humidifier1[0]) + abs(j - humidifier1[1]) <= D:
                            humidified_set.add((i, j))
            
            # Check cells humidified by the second humidifier
            for i in range(H):
                for j in range(W):
                    if grid[i][j] == '.':
                        if abs(i - humidifier2[0]) + abs(j - humidifier2[1]) <= D:
                            humidified_set.add((i, j))
            
            # Update the maximum humidified cells count
            max_humidified = max(max_humidified, len(humidified_set))
    
    print(max_humidified)