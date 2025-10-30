# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
M = int(data[1])
A = list(map(int, data[2:]))

A.sort()

if sum(A) <= M:
    print("infinite")
else:
    left, right = 0, max(A)
    while left < right:
        mid = (left + right + 1) // 2
        total = sum(min(mid, a) for a in A)
        if total <= M:
            left = mid
        else:
            right = mid - 1
    print(left)