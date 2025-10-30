import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
people = []

for i in range(N):
    A_i = int(data[2 * i + 1])
    B_i = int(data[2 * i + 2])
    success_rate = A_i / (A_i + B_i)
    people.append((success_rate, i + 1))

people.sort(key=lambda x: (-x[0], x[1]))

for person in people:
    print(person[1], end=' ')