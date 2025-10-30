# YOUR CODE HERE
n, m = map(int, input().split())
x = list(map(int, input().split()))
a = list(map(int, input().split()))

# Check if total stones equals n
total_stones = sum(a)
if total_stones != n:
    print(-1)
else:
    # Create list of (position, stones) and sort by position
    stones = list(zip(x, a))
    stones.sort()
    
    operations = 0
    target_pos = 1  # Next position that needs to be filled
    
    for pos, count in stones:
        # Check if we can reach the required target positions
        if target_pos < pos:
            # We have gaps that can't be filled (no stones can move backwards)
            print(-1)
            exit()
        
        # Assign stones from this position to target positions
        for i in range(count):
            current_target = target_pos + i
            if current_target > n:
                print(-1)
                exit()
            
            # Cost to move stone from pos to current_target
            operations += max(0, current_target - pos)
        
        target_pos += count
    
    print(operations)