# YOUR CODE HERE
import sys
import heapq

input = sys.stdin.read
data = input().split()

N = int(data[0])
M = int(data[1])

wheels = []
index = 2
for _ in range(N):
    C = int(data[index])
    P = int(data[index + 1])
    S = list(map(int, data[index + 2:index + 2 + P]))
    wheels.append((C, P, S))
    index += 2 + P

# Calculate the expected value for each wheel
expected_values = []
for C, P, S in wheels:
    expected_points = sum(S) / P
    expected_cost = C / expected_points
    expected_values.append((expected_cost, expected_points, C))

# Use a priority queue to always pick the wheel with the lowest expected cost per point
heapq.heapify(expected_values)

total_cost = 0
points_earned = 0

while points_earned < M:
    expected_cost, expected_points, C = heapq.heappop(expected_values)
    plays_needed = (M - points_earned + expected_points - 1) // expected_points
    total_cost += plays_needed * C
    points_earned += plays_needed * expected_points

print(f"{total_cost:.15f}")