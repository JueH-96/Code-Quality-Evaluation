n = int(input())
people = []
for i in range(n):
    s, a = input().split()
    a = int(a)
    people.append((a, s, i + 1))  # Storing age, name, and original position

# Find the youngest person
youngest = min(people, key=lambda x: x[0])
start_index = youngest[2] - 1  # Convert to 0-based index

# Generate the order starting from the youngest
order = people[start_index:] + people[:start_index]

# Print each name in order
for person in order:
    print(person[1])