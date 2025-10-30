N = int(input())
employees = [0] * 24
for _ in range(N):
    W, X = map(int, input().split())
    for i in range(9):
        employees[(24 - X + i) % 24] += W
max_employees = 0
for i in range(24):
    max_employees = max(max_employees, employees[i])
print(max_employees)