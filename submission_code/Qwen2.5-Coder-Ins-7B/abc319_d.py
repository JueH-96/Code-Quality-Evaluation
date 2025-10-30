# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()
N = int(data[0])
M = int(data[1])
L = list(map(int, data[2:]))

def can_fit(width):
    lines = 1
    current_width = 0
    for word in L:
        if current_width + word + 1 > width:
            lines += 1
            current_width = word
        else:
            current_width += word + 1
    return lines <= M

left, right = max(max(L), sum(L) - M + 1), sum(L)
while left < right:
    mid = (left + right) // 2
    if can_fit(mid):
        right = mid
    else:
        left = mid + 1
print(left)