# YOUR CODE HERE
n, m = map(int, input().split())
h = list(map(int, input().split()))
count = 0
for i in h:
    if m >= i:
        m -= i
        count += 1
    else:
        break
print(count)