# YOUR CODE HERE
n = int(input())
a = list(map(int, input().split()))
b = []
for i in range(n):
    start = i * 7
    end = start + 7
    week_sum = sum(a[start:end])
    b.append(week_sum)
print(' '.join(map(str, b)))