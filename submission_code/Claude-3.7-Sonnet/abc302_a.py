def min_attacks_required(stamina, damage):
    # Calculate minimum number of attacks needed to make stamina 0 or less
    if stamina % damage == 0:
        return stamina // damage
    else:
        return stamina // damage + 1

# Read input
A, B = map(int, input().split())

# Calculate and print the result
print(min_attacks_required(A, B))