# YOUR CODE HERE
n = int(input())
people = []
for i in range(n):
    line = input().split()
    name = line[0]
    age = int(line[1])
    people.append((name, age))

# Find the index of the youngest person
min_age = float('inf')
min_idx = -1
for i in range(n):
    if people[i][1] < min_age:
        min_age = people[i][1]
        min_idx = i

# Print names starting from the youngest person in clockwise order
for i in range(n):
    idx = (min_idx + i) % n
    print(people[idx][0])