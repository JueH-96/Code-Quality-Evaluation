# YOUR CODE HERE
n = int(input())
people = []
for i in range(n):
    line = input().split()
    name = line[0]
    age = int(line[1])
    people.append((name, age, i))

# Find the person with minimum age
min_age_person = min(people, key=lambda x: x[1])
start_index = min_age_person[2]

# Print names starting from the youngest person in clockwise order
for i in range(n):
    current_index = (start_index + i) % n
    print(people[current_index][0])