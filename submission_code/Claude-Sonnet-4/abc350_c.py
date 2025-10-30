# YOUR CODE HERE
n = int(input())
a = list(map(int, input().split()))

# Convert to 1-indexed for easier reasoning
# a[0] corresponds to position 1, a[1] to position 2, etc.

operations = []

for i in range(n):
    # We want a[i] to equal i+1
    if a[i] != i + 1:
        # Find where i+1 is located
        target_pos = -1
        for j in range(i + 1, n):
            if a[j] == i + 1:
                target_pos = j
                break
        
        # Swap a[i] and a[target_pos]
        a[i], a[target_pos] = a[target_pos], a[i]
        # Record the operation (convert to 1-indexed positions)
        operations.append((i + 1, target_pos + 1))

print(len(operations))
for op in operations:
    print(op[0], op[1])