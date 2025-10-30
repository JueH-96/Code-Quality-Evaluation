# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
people = []

for i in range(N):
    A = int(data[2*i+1])
    B = int(data[2*i+2])
    success_rate = A / (A + B)
    people.append((i+1, success_rate))

people.sort(key=lambda x: (-x[1], x[0]))

for person in people:
    print(person[0], end=" ")