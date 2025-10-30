n = int(input())
people = []
for i in range(n):
    s, a = input().split()
    a = int(a)
    pos = i + 1
    people.append((s, a, pos))

# Find the youngest person's position
youngest = min(people, key=lambda x: x[1])
start_pos = youngest[2]

# Generate the rotated list of positions
original_order = list(range(1, n + 1))
index = original_order.index(start_pos)
rotated = original_order[index:] + original_order[:index]

# Create a dictionary to map position to name
pos_to_name = {pos: name for name, _, pos in people}

# Output the names in the rotated order
for pos in rotated:
    print(pos_to_name[pos])