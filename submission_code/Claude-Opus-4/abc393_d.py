# YOUR CODE HERE
n = int(input())
s = input().strip()

# Find all positions of 1s
ones_positions = []
for i in range(n):
    if s[i] == '1':
        ones_positions.append(i)

k = len(ones_positions)
if k <= 1:
    print(0)
else:
    min_cost = float('inf')
    
    # Try each possible starting position for the contiguous block
    for start in range(n - k + 1):
        cost = 0
        # Calculate cost to move each 1 to its target position
        for i in range(k):
            cost += abs(ones_positions[i] - (start + i))
        
        min_cost = min(min_cost, cost)
    
    print(min_cost)