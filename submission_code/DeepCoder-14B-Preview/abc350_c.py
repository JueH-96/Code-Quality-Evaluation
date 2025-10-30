n = int(input())
a = list(map(int, input().split()))
pos = [0] * (n + 1)  # Using 1-based indexing for values

for i in range(n):
    x = a[i]
    pos[x] = i  # Store the zero-based index where each value is found

swaps = []

for i in range(n):
    if a[i] == i + 1:
        continue
    # Find the position of the element that should be at i
    x = i + 1
    j = pos[x]
    # Perform the swap
    old_i_val = a[i]
    old_j_val = a[j]
    a[i], a[j] = a[j], a[i]
    swaps.append((i + 1, j + 1))  # Convert to 1-based indices for output
    # Update the position array
    pos[old_i_val] = j
    pos[old_j_val] = i

# Output the results
print(len(swaps))
for swap in swaps:
    print(swap[0], swap[1])