# YOUR CODE HERE
def min_operations_to_k_consecutive_o(H, W, K, grid):
    min_operations = float('inf')
    
    # Check horizontally
    for i in range(H):
        for j in range(W - K + 1):
            # Count the number of '.' in this segment
            count_dots = 0
            for l in range(K):
                if grid[i][j + l] == '.':
                    count_dots += 1
                elif grid[i][j + l] == 'x':
                    count_dots = float('inf')
                    break
            min_operations = min(min_operations, count_dots)
    
    # Check vertically
    for j in range(W):
        for i in range(H - K + 1):
            # Count the number of '.' in this segment
            count_dots = 0
            for l in range(K):
                if grid[i + l][j] == '.':
                    count_dots += 1
                elif grid[i + l][j] == 'x':
                    count_dots = float('inf')
                    break
            min_operations = min(min_operations, count_dots)
    
    # If min_operations is still inf, it means no valid sequence was found
    return min_operations if min_operations != float('inf') else -1

if __name__ == "__main__":
    import sys
    input = sys.stdin.read
    data = input().split()
    
    H = int(data[0])
    W = int(data[1])
    K = int(data[2])
    grid = data[3:3+H]
    
    result = min_operations_to_k_consecutive_o(H, W, K, grid)
    print(result)