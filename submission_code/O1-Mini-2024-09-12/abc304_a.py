n = int(input())
people = []
for _ in range(n):
    name, age = input().split()
    age = int(age)
    people.append((name, age))

# Find the index of the youngest person
min_age = float('inf')
start_idx = 0
for idx, (name, age) in enumerate(people):
    if age < min_age:
        min_age = age
        start_idx = idx

# Print names starting from the youngest person in clockwise order
for i in range(n):
    idx = (start_idx + i) % n
    print(people[idx][0])