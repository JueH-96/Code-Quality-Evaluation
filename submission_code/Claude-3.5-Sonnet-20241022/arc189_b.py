def get_min_sum(N, X):
    # Sort X in ascending order and store original indices
    sorted_pairs = [(x, i) for i, x in enumerate(X)]
    sorted_pairs.sort()
    
    # Function to perform one operation
    def perform_operation(coords, i):
        # Calculate midpoint between i and i+3
        M = (coords[i] + coords[i+3]) / 2
        
        # Move i+1 and i+2 to symmetric positions around M
        new_coords = coords.copy()
        new_coords[i+1] = 2*M - coords[i+2]  # Symmetric point for i+1
        new_coords[i+2] = 2*M - coords[i+1]  # Symmetric point for i+2
        
        return new_coords

    def try_all_operations(coords):
        min_sum = sum(coords)
        changed = True
        
        while changed:
            changed = False
            for i in range(N-3):
                new_coords = perform_operation(coords, i)
                new_sum = sum(new_coords)
                
                if new_sum < min_sum:
                    min_sum = new_sum
                    coords = new_coords
                    changed = True
                    
        return min_sum

    # Extract just the coordinates in sorted order
    coords = [x for x, _ in sorted_pairs]
    
    # Try operations until no improvement is possible
    result = try_all_operations(coords)
    
    return int(result)

# Read input
N = int(input())
X = list(map(int, input().split()))

# Calculate and print result
print(get_min_sum(N, X))