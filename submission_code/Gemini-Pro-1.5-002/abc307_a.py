# YOUR CODE HERE
n = int(input())
a = list(map(int, input().split()))
b = []
for i in range(n):
    week_sum = 0
    for j in range(7):
        week_sum += a[i * 7 + j]
    b.append(week_sum)
print(*b)