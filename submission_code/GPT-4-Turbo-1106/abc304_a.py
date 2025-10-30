# Read the number of people
N = int(input().strip())

# Read the names and ages of the people and store them in a list
people = []
for _ in range(N):
    name, age = input().strip().split()
    age = int(age)
    people.append((name, age))

# Sort the people by age to find the youngest person
people.sort(key=lambda x: x[1])

# Find the index of the youngest person
youngest_index = 0

# Print the names starting from the youngest person in clockwise order
for i in range(N):
    print(people[(youngest_index + i) % N][0])