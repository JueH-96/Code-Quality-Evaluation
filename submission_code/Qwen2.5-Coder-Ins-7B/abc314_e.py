import sys
import math

input = sys.stdin.read
data = input().split()

N = int(data[0])
M = int(data[1])

wheels = []
index = 2
for i in range(N):
    C = int(data[index])
    P = int(data[index + 1])
    S = list(map(int, data[index + 2:index + 2 + P]))
    wheels.append((C, P, S))
    index += 2 + P

# Calculate the expected value for each wheel
expected_values = []
for C, P, S in wheels:
    expected_value = sum(S) / P
    expected_values.append((C, expected_value))

# Sort wheels by their expected value per yen spent
expected_values.sort(key=lambda x: x[1] / x[0], reverse=True)

# Calculate the expected total cost
total_cost = 0
points = 0
for C, expected_value in expected_values:
    while points < M:
        points += expected_value
        total_cost += C

print(total_cost)