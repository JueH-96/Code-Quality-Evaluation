import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
M = int(data[1])
L = list(map(int, data[2:]))

def can_fit(width):
    lines = 1
    current_width = 0
    for length in L:
        if current_width + length > width:
            lines += 1
            current_width = length
            if lines > M:
                return False
        else:
            current_width += length + 1
    return True

left, right = max(L), sum(L) + N - 1
while left < right:
    mid = (left + right) // 2
    if can_fit(mid):
        right = mid
    else:
        left = mid + 1

print(left)