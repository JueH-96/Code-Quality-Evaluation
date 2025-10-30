n = int(input())
sum_a = 0
max_diff = -float('inf')

for _ in range(n):
    a, b = map(int, input().split())
    sum_a += a
    diff = b - a
    if diff > max_diff:
        max_diff = diff

print(sum_a + max_diff)