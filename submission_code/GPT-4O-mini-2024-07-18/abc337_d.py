def min_operations_to_form_sequence(H, W, K, grid):
    # Function to calculate the minimum operations for a horizontal sequence
    def min_operations_horizontal():
        min_ops = float('inf')
        for i in range(H):
            # Count the number of '.' in each window of size K
            current_ops = 0
            for j in range(K):
                if grid[i][j] == '.':
                    current_ops += 1
            
            if current_ops < min_ops:
                min_ops = current_ops
            
            for j in range(K, W):
                if grid[i][j - K] == '.':
                    current_ops -= 1
                if grid[i][j] == '.':
                    current_ops += 1
                
                if current_ops < min_ops:
                    min_ops = current_ops
        
        return min_ops
    
    # Function to calculate the minimum operations for a vertical sequence
    def min_operations_vertical():
        min_ops = float('inf')
        for j in range(W):
            current_ops = 0
            for i in range(K):
                if grid[i][j] == '.':
                    current_ops += 1
            
            if current_ops < min_ops:
                min_ops = current_ops
            
            for i in range(K, H):
                if grid[i - K][j] == '.':
                    current_ops -= 1
                if grid[i][j] == '.':
                    current_ops += 1
                
                if current_ops < min_ops:
                    min_ops = current_ops
        
        return min_ops
    
    # Calculate minimum operations for both orientations
    min_horizontal = min_operations_horizontal()
    min_vertical = min_operations_vertical()
    
    # Determine the overall minimum operations needed
    overall_min = min(min_horizontal, min_vertical)
    
    # If no valid sequence can be formed, return -1
    if overall_min == float('inf'):
        return -1
    
    return overall_min

# Read input
import sys
input = sys.stdin.read
data = input().splitlines()

H, W, K = map(int, data[0].split())
grid = data[1:H + 1]

# Get the result and print it
result = min_operations_to_form_sequence(H, W, K, grid)
print(result)