import sys

input = sys.stdin.read
data = input().split()

N = int(data[0])
people = []

for i in range(N):
    name = data[2 * i + 1]
    age = int(data[2 * i + 2])
    people.append((name, age))

# Find the youngest person
youngest_index = min(range(N), key=lambda i: people[i][1])

# Print names starting from the youngest person
for i in range(N):
    print(people[(youngest_index + i) % N][0])