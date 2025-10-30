# YOUR CODE HERE
N = int(input())

# Iterate through x from 0 to N (inclusive)
for x in range(N + 1):
    # For each x, iterate through y from 0 to N - x (inclusive)
    # This ensures x + y <= N
    for y in range(N - x + 1):
        # For each pair (x, y), iterate through z from 0 to N - x - y (inclusive)
        # This ensures x + y + z <= N
        for z in range(N - x - y + 1):
            # Print the triple (x, y, z)
            # The loops generate triples in lexicographical order automatically.
            print(f"{x} {y} {z}")