n, m = map(int, input().split())
a = list(map(int, input().split()))
sums = [0] * m

for _ in range(n):
    x = list(map(int, input().split()))
    for j in range(m):
        sums[j] += x[j]

result = 'Yes'
for j in range(m):
    if sums[j] < a[j]:
        result = 'No'
        break

print(result)