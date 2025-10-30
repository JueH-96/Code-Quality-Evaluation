n = int(input())
giants = []
total_a = 0

for i in range(n):
    a, b = map(int, input().split())
    giants.append((a, b))
    total_a += a

max_diff = float('-inf')
for a, b in giants:
    diff = b - a
    if diff > max_diff:
        max_diff = diff

result = total_a + max_diff
print(result)