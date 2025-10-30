def min_operations():
    n, m = map(int, input().split())
    positions = list(map(int, input().split()))
    stones = list(map(int, input().split()))
    
    # Check if the total number of stones equals N
    total_stones = sum(stones)
    if total_stones != n:
        return -1
    
    # Create a mapping of positions to number of stones
    stones_at_pos = {}
    for i in range(m):
        stones_at_pos[positions[i]] = stones[i]
    
    operations = 0
    excess_stones = 0
    
    # We need to iterate through all positions from 1 to n
    # For positions with no stones, we need to bring stones from the left (if available)
    for i in range(1, n + 1):
        # Add stones at the current position to our available stones
        excess_stones += stones_at_pos.get(i, 0)
        
        # Check if we have enough stones to place one at the current position
        if excess_stones < 1:
            return -1  # Not enough stones to fill this position
        
        # Place one stone at the current position
        excess_stones -= 1
        
        # Calculate operations to move the excess stones to the right
        # Each excess stone at position i requires 1 operation to move to position i+1
        operations += excess_stones
    
    return operations

print(min_operations())