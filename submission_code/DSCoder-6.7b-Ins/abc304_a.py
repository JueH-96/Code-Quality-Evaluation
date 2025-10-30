# YOUR CODE HERE

import sys

N = int(sys.stdin.readline().strip())
people = []
for i in range(N):
    name, age = sys.stdin.readline().strip().split()
    people.append((name, int(age)))

people.sort(key=lambda x: x[1])

youngest_index = people.index(min(people, key=lambda x: x[1]))

for i in range(N):
    print(people[(youngest_index + i) % N][0])