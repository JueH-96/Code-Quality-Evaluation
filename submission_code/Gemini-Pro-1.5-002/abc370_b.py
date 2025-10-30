# YOUR CODE HERE
n = int(input())
a = []
for _ in range(n):
    a.append(list(map(int, input().split())))

current = 1
for i in range(1, n + 1):
    if current >= i:
        current = a[current - 1][i - 1]
    else:
        current = a[i - 1][current - 1]

print(current)