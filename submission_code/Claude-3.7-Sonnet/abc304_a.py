# Read the number of people
N = int(input())
people = []

# Read the information for each person
for i in range(N):
    name, age = input().split()
    people.append((name, int(age)))

# Find the index of the youngest person
youngest_index = min(range(N), key=lambda i: people[i][1])

# Print the names in clockwise order starting from the youngest person
for i in range(N):
    index = (youngest_index + i) % N
    print(people[index][0])