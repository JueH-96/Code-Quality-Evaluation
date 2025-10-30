# YOUR CODE HERE
n, m = map(int, input().split())
a = list(map(int, input().split()))
x = []
for _ in range(n):
    x.append(list(map(int, input().split())))

nutrients = [0] * m
for i in range(n):
    for j in range(m):
        nutrients[j] += x[i][j]

result = 'Yes'
for i in range(m):
    if nutrients[i] < a[i]:
        result = 'No'
        break

print(result)