# YOUR CODE HERE
n = int(input())
a = list(map(int, input().split()))

total = 0
for i in range(n):
    for j in range(i + 1, n):
        total += max(a[j] - a[i], 0)

print(total)