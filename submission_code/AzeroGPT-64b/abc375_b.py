import math
N = int(input())
prev_point = (0, 0)
total_cost = 0
for _ in range(N):
    Xi, Yi = map(int, input().split())
    total_cost += math.sqrt((prev_point[0] - Xi) ** 2 + (prev_point[1] - Yi) ** 2)
    prev_point = (Xi, Yi)
total_cost += math.sqrt((prev_point[0] - 0) ** 2 + (prev_point[1] - 0) ** 2)
print(total_cost)