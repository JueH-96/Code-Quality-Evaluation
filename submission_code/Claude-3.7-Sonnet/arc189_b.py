def min_coordinates_sum(N, X):
    positions = X.copy()
    
    while True:
        best_i = -1
        best_change = 0
        
        for i in range(N - 3):  # i is 0-indexed, so we go from 0 to N-4
            # Calculate midpoint between positions i and i+3
            midpoint = (positions[i] + positions[i+3]) / 2
            
            # Calculate new positions for i+1 and i+2 after the operation
            new_pos_i_plus_1 = 2 * midpoint - positions[i+1]
            new_pos_i_plus_2 = 2 * midpoint - positions[i+2]
            
            # Calculate the change in the sum if we perform this operation
            change = (new_pos_i_plus_1 + new_pos_i_plus_2) - (positions[i+1] + positions[i+2])
            
            if change < best_change:
                best_change = change
                best_i = i
        
        if best_i == -1 or best_change >= 0:
            break
        
        # Perform the operation for the best i
        midpoint = (positions[best_i] + positions[best_i+3]) / 2
        positions[best_i+1] = 2 * midpoint - positions[best_i+1]
        positions[best_i+2] = 2 * midpoint - positions[best_i+2]
        
        # Sort the positions to maintain ascending order
        positions.sort()
    
    return sum(positions)

# Read input
N = int(input())
X = list(map(int, input().split()))

# Solve and print the result
print(min_coordinates_sum(N, X))