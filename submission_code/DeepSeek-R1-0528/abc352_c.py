n = int(input().strip())
total_A = 0
max_diff = 0
for _ in range(n):
    a, b = map(int, input().split())
    total_A += a
    diff = b - a
    if diff > max_diff:
        max_diff = diff
print(total_A + max_diff)