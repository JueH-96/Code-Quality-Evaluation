n = int(input())
sum_A = 0
max_diff = 0
for _ in range(n):
    A, B = map(int, input().split())
    sum_A += A
    diff = B - A
    if diff > max_diff:
        max_diff = diff
print(sum_A + max_diff)