n = int(input())
sum_a = 0
max_diff = -float('inf')
for _ in range(n):
    a, b = map(int, input().split())
    sum_a += a
    current_diff = b - a
    if current_diff > max_diff:
        max_diff = current_diff
print(sum_a + max_diff)