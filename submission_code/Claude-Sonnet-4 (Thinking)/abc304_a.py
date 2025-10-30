n = int(input())
people = []
for i in range(n):
    parts = input().split()
    name = parts[0]
    age = int(parts[1])
    people.append((name, age))

# Find the position of the youngest person
min_age = min(age for name, age in people)
youngest_pos = None
for i, (name, age) in enumerate(people):
    if age == min_age:
        youngest_pos = i
        break

# Starting from the youngest person, print names in clockwise order
for i in range(n):
    pos = (youngest_pos + i) % n
    print(people[pos][0])