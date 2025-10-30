# YOUR CODE HERE
n = int(input())
p = list(map(int, input().split()))

# Convert to 0-indexed for easier calculation
p = [x - 1 for x in p]

# Count inversions for each element
# inversions[i] = number of elements to the left of position i that are greater than p[i]
inversions = [0] * n
for i in range(n):
    for j in range(i):
        if p[j] > p[i]:
            inversions[i] += 1

# The total cost is the sum of costs for moving each element to its correct position
# When an element at position i needs to move left by k positions,
# it will use swap positions (i-k), (i-k+1), ..., (i-1)
# But we need to be careful about the order of operations

# Actually, let's think differently:
# For each position i, count how many times a swap at position i is needed
# This happens when we need to move an element from right to left through position i

# Count how many times each position is used for swapping
swap_count = [0] * (n - 1)

for i in range(n):
    # Element p[i] needs to be at position p[i]
    target = p[i]
    current = i
    
    # Count how many positions to the left this element needs to move
    if target < current:
        # This element needs to move left from position current to position target
        # It will use swap positions target, target+1, ..., current-1
        for j in range(target, current):
            swap_count[j] += 1

# Calculate total cost
total_cost = 0
for i in range(n - 1):
    # Swap at position i costs (i+1) in 1-indexed
    total_cost += swap_count[i] * (i + 1)

print(total_cost)