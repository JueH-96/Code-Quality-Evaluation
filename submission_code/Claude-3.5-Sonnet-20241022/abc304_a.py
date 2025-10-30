N = int(input())
people = []
for i in range(N):
    name, age = input().split()
    age = int(age)
    people.append((name, age, i+1))

# Find youngest person
youngest_idx = min(range(N), key=lambda i: people[i][1])

# Reorder list starting from youngest person
result = []
for i in range(N):
    curr_idx = (youngest_idx + i) % N
    result.append(people[curr_idx][0])

# Print names in order
for name in result:
    print(name)