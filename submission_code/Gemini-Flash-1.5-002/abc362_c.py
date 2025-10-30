# YOUR CODE HERE
n = int(input())
ranges = []
for i in range(n):
    l, r = map(int, input().split())
    ranges.append((l, r))

total_min = sum(l for l, r in ranges)
total_max = sum(r for l, r in ranges)

if total_max < 0 or total_min > 0:
    print('No')
else:
    x = [0] * n
    current_sum = 0
    for i in range(n - 1):
        x[i] = ranges[i][0]
        current_sum += x[i]
    x[n - 1] = -current_sum
    if total_min <= 0 <= total_max and ranges[n-1][0] <= x[n-1] <= ranges[n-1][1]:
        print('Yes')
        print(*x)
    else:
        print('No')