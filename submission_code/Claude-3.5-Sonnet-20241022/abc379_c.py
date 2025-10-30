def solve():
    # Read input
    N, M = map(int, input().split())
    X = list(map(int, input().split()))
    A = list(map(int, input().split()))
    
    # First check: total stones should equal N
    total_stones = sum(A)
    if total_stones != N:
        return -1
    
    # Create initial state array
    stones = [0] * (N + 1)  # 1-based indexing
    for i in range(M):
        stones[X[i]] = A[i]
    
    operations = 0
    excess = 0  # Stones that need to move right
    
    # Process from left to right
    for i in range(1, N + 1):
        stones_here = stones[i] + excess
        
        # We need exactly one stone in each cell
        if stones_here < 1:
            # Not enough stones to put one here
            return -1
            
        excess = stones_here - 1  # Move extra stones to the right
        if excess > 0:
            operations += excess  # Count moves needed
    
    # After the loop, excess should be 0
    if excess != 0:
        return -1
        
    return operations

# Print the answer
print(solve())