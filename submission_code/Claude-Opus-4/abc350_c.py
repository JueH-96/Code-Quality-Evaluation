# YOUR CODE HERE
n = int(input())
a = list(map(int, input().split()))

# Convert to 0-indexed for easier manipulation
a = [x - 1 for x in a]

operations = []

# For each position i, put the correct element (i) in position i
for i in range(n):
    # Find where element i is currently located
    j = i
    while j < n and a[j] != i:
        j += 1
    
    # If element i is not at position i, swap it
    if j != i:
        a[i], a[j] = a[j], a[i]
        # Store the operation (convert back to 1-indexed)
        operations.append((i + 1, j + 1))

# Output the result
print(len(operations))
for op in operations:
    print(op[0], op[1])