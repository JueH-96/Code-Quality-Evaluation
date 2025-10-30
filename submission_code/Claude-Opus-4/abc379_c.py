# YOUR CODE HERE
N, M = map(int, input().split())
X = list(map(int, input().split()))
A = list(map(int, input().split()))

# Create a list of (position, count) pairs and sort by position
stones = []
for i in range(M):
    stones.append((X[i], A[i]))
stones.sort()

# Check if total stones equals N
total_stones = sum(A)
if total_stones != N:
    print(-1)
else:
    # Calculate minimum operations
    operations = 0
    current_stone = 1  # The position where the next stone should go
    
    for pos, count in stones:
        # For each stone at position 'pos', calculate where it should go
        for _ in range(count):
            # This stone should go to position 'current_stone'
            if pos > current_stone:
                # Stone is already past where it should be - impossible
                operations = -1
                break
            operations += current_stone - pos
            current_stone += 1
        
        if operations == -1:
            break
    
    print(operations)