n = int(input())
total_a = 0
max_diff = float('-inf')

for i in range(n):
    a, b = map(int, input().split())
    total_a += a
    max_diff = max(max_diff, b - a)

print(total_a + max_diff)