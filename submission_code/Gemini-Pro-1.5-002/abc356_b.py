# YOUR CODE HERE
n, m = map(int, input().split())
a = list(map(int, input().split()))
x = []
for _ in range(n):
    x.append(list(map(int, input().split())))

totals = [0] * m
for j in range(m):
    for i in range(n):
        totals[j] += x[i][j]

for j in range(m):
    if totals[j] < a[j]:
        print("No")
        exit()

print("Yes")