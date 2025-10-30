# Read input
N = int(input())
people = []
for _ in range(N):
    s, a = input().split()
    people.append((s, int(a)))

# Find the index of the youngest person
min_age = float('inf')
min_index = -1
for i in range(N):
    if people[i][1] < min_age:
        min_age = people[i][1]
        min_index = i

# Generate and print the output order
for i in range(N):
    idx = (min_index + i) % N
    print(people[idx][0])