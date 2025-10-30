# YOUR CODE HERE
N = int(input())
ranges = []
for _ in range(N):
    L, R = map(int, input().split())
    ranges.append((L, R))

# Calculate the minimum and maximum possible sums
min_sum = sum(L for L, R in ranges)
max_sum = sum(R for L, R in ranges)

# Check if 0 is within the possible range
if min_sum > 0 or max_sum < 0:
    print("No")
else:
    # Start with all minimum values
    X = [L for L, R in ranges]
    current_sum = min_sum
    
    # We need to add (0 - current_sum) to make the sum 0
    needed = -current_sum
    
    # Distribute the needed amount
    for i in range(N):
        L, R = ranges[i]
        # Maximum we can add to X[i]
        can_add = R - L
        # Add as much as possible
        add_amount = min(needed, can_add)
        X[i] += add_amount
        needed -= add_amount
        
        if needed == 0:
            break
    
    print("Yes")
    print(*X)