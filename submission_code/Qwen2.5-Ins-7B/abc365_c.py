# YOUR CODE HERE
n, m = map(int, input().split())
a = list(map(int, input().split()))

if sum(a) <= m:
    print('infinite')
else:
    left, right = 0, max(a)
    while left < right:
        mid = (left + right + 1) // 2
        if sum(min(mid, ai) for ai in a) <= m:
            left = mid
        else:
            right = mid - 1
    print(left)