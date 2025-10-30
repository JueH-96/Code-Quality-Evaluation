import sys
from collections import deque

input = sys.stdin.read
data = input().split()

N = int(data[0])
A = list(map(int, data[1:]))

# Initialize the result list
result = [0] * N

# Process left to right
stack = deque()
for i in range(N):
    while stack and stack[-1] < A[i]:
        stack.pop()
    stack.append(A[i])
    result[i] = stack[0]

# Process right to left
stack = deque()
for i in range(N-1, -1, -1):
    while stack and stack[-1] < A[i]:
        stack.pop()
    stack.append(A[i])
    result[i] = max(result[i], stack[0])

print(*result)