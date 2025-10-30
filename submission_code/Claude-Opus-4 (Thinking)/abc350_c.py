n = int(input())
a = list(map(int, input().split()))

operations = []

# Create a position map for efficiency
pos = [0] * (n + 1)
for j in range(n):
    pos[a[j]] = j

# For each position i (0-indexed), place the correct element (i+1) there
for i in range(n):
    # Element (i+1) should be at position i
    if a[i] != i + 1:
        # Find where element (i+1) is currently located
        current_pos = pos[i + 1]
        
        # Update position map
        pos[a[i]] = current_pos
        pos[i + 1] = i
        
        # Swap elements
        a[i], a[current_pos] = a[current_pos], a[i]
        
        # Record the operation (1-indexed)
        operations.append((i + 1, current_pos + 1))

# Output
print(len(operations))
for op in operations:
    print(op[0], op[1])