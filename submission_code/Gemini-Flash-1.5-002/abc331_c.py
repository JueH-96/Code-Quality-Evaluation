# YOUR CODE HERE
n = int(input())
a = list(map(int, input().split()))

results = []
for i in range(n):
    sum_greater = 0
    for j in range(n):
        if a[j] > a[i]:
            sum_greater += a[j]
    results.append(sum_greater)

print(*results)