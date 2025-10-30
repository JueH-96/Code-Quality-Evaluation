# Read the number of people
N = int(input())

# Read the names and ages of each person
people = []
for _ in range(N):
    S, A = input().split()
    A = int(A)
    people.append({'name': S, 'age': A})

# Find the youngest person
youngest = min(people, key=lambda x: x['age'])

# Find the index of the youngest person
index = people.index(youngest)

# Print the names in clockwise order starting from the youngest
for i in range(N):
    print(people[(index + i) % N]['name'])