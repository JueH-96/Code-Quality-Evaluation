import sys

N, M = map(int, input().split())
L = list(map(int, input().split()))

left, right = 0, sum(L)
while left <= right:
    mid = (left + right) // 2
    lines = 1
    curr_width = 0
    for l in L:
        if curr_width + l > mid:
            lines += 1
            curr_width = l
        else:
            curr_width += l + 1
    if lines <= M:
        right = mid - 1
    else:
        left = mid + 1

print(left)