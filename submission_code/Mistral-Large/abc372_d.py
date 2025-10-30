import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
H = list(map(int, data[1:]))

stack = []
answer = [0] * N

for i in range(N):
    while stack and H[stack[-1]] < H[i]:
        index = stack.pop()
        answer[index] = i - index - 1
    stack.append(i)

while stack:
    index = stack.pop()
    answer[index] = N - index - 1

print(" ".join(map(str, answer)))