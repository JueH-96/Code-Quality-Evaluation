# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
employees = []

for i in range(N):
    W = int(data[2*i+1])
    X = int(data[2*i+2])
    start = (X + 15) % 24
    end = (X + 24) % 24
    employees.append((W, start, end))

employees.sort(key=lambda x: (x[1], x[2]))

max_employees = 0
current_employees = 0
i = 0

for start in range(9, 19):
    while i < N and employees[i][1] <= start:
        current_employees += employees[i][0]
        i += 1
    max_employees = max(max_employees, current_employees)
    while i < N and employees[i][2] <= start:
        current_employees -= employees[i][0]
        i += 1

print(max_employees)