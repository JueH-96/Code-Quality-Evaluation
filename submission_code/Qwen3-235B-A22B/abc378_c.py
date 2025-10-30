# Read input
n = int(input())
a = list(map(int, input().split()))

# Dictionary to keep track of last positions
last_pos = {}

# Result list
b = []

# Iterate through each element in the list
for i in range(n):
    val = a[i]
    # Check if the value was seen before
    if val in last_pos:
        b.append(last_pos[val])
    else:
        b.append(-1)
    # Update the last occurrence of the value to current (1-based) index
    last_pos[val] = i + 1  # i starts at 0, so i+1 gives 1-based position

# Print the result
print(' '.join(map(str, b)))