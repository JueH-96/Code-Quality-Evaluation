# YOUR CODE HERE
n = int(input())
people = []

for i in range(n):
    line = input().split()
    name = line[0]
    age = int(line[1])
    people.append((name, age, i))  # store name, age, and original position

# Find the youngest person
youngest_age = min(people, key=lambda x: x[1])[1]
youngest_pos = -1

for i in range(n):
    if people[i][1] == youngest_age:
        youngest_pos = i
        break

# Print names starting from youngest person's position in clockwise order
for i in range(n):
    pos = (youngest_pos + i) % n
    print(people[pos][0])