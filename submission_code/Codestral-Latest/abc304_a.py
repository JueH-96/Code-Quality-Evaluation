# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
people = []

for i in range(1, 2*N, 2):
    name = data[i]
    age = int(data[i+1])
    people.append((age, name))

# Sort people by age
people.sort()

# Find the starting index of the youngest person
start_index = 0
for i in range(N):
    if people[i][0] == min(age for age, _ in people):
        start_index = i
        break

# Print names in clockwise order starting from the youngest person
for i in range(start_index, N):
    print(people[i][1])
for i in range(start_index):
    print(people[i][1])