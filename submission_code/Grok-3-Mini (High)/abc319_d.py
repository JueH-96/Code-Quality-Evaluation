import sys
data = sys.stdin.read().split()
N = int(data[0])
M = int(data[1])
L = [int(x) for x in data[2:2+N]]

def num_lines(W):
    lines = 1
    curr_width = L[0]
    for width in L[1:]:
        if curr_width + 1 + width <= W:
            curr_width += 1 + width
        else:
            lines += 1
            curr_width = width
    return lines

max_L = max(L)
sum_L = sum(L)
left = max_L
right = sum_L + (N - 1)

while left <= right:
    mid = (left + right) // 2
    if num_lines(mid) <= M:
        right = mid - 1
    else:
        left = mid + 1

print(left)