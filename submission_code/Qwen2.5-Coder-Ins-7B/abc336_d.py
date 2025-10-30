# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
A = list(map(int, data[1:]))

left = 0
right = N - 1
max_size = 0

while left <= right:
    if A[left] == A[right]:
        max_size += 1
        left += 1
        right -= 1
    elif A[left] < A[right]:
        left += 1
    else:
        right -= 1

print(max_size)