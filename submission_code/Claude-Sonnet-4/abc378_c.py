# YOUR CODE HERE
n = int(input())
a = list(map(int, input().split()))

# Dictionary to store the most recent position of each value
last_pos = {}
b = []

for i in range(n):
    value = a[i]
    
    if value in last_pos:
        # Found a previous occurrence
        b.append(last_pos[value])
    else:
        # No previous occurrence
        b.append(-1)
    
    # Update the most recent position for this value (1-indexed)
    last_pos[value] = i + 1

print(' '.join(map(str, b)))