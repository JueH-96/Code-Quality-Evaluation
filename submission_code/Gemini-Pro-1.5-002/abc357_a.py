# YOUR CODE HERE
n, m = map(int, input().split())
h = list(map(int, input().split()))

count = 0
for i in range(n):
    if m >= h[i]:
        m -= h[i]
        count += 1
    else:
        break

print(count)