# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
H = list(map(int, data[1:]))

result = []
stack = []

for i in range(N):
    count = 0
    while stack and H[stack[-1]] <= H[i]:
        stack.pop()
        count += 1
    if stack:
        count += 1
    result.append(count)
    stack.append(i)

print(' '.join(map(str, result)))