# YOUR CODE HERE
n, m = map(int, input().split())
l = list(map(int, input().split()))

def can_fit(w):
    lines = 1
    width = 0
    for i in range(n):
        if width + l[i] + (1 if width > 0 else 0) > w:
            lines += 1
            width = l[i]
        else:
            width += l[i] + (1 if width > 0 else 0)
    return lines <= m

low, high = max(l), sum(l) + n - 1
while low < high:
    mid = (low + high) // 2
    if can_fit(mid):
        high = mid
    else:
        low = mid + 1

print(low)