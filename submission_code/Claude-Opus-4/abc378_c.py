# YOUR CODE HERE
n = int(input())
a = list(map(int, input().split()))

# Dictionary to store the most recent position of each value
last_position = {}
result = []

for i in range(n):
    current_value = a[i]
    
    # Check if we've seen this value before
    if current_value in last_position:
        # Get the most recent position (1-indexed)
        result.append(last_position[current_value])
    else:
        # Haven't seen this value before
        result.append(-1)
    
    # Update the most recent position for this value (1-indexed)
    last_position[current_value] = i + 1

print(' '.join(map(str, result)))