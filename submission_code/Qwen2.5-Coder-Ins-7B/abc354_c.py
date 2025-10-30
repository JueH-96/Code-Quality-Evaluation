# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
cards = []

for i in range(1, 2 * N, 2):
    A = int(data[i])
    C = int(data[i + 1])
    cards.append((A, C))

cards.sort()

stack = []
for A, C in cards:
    while stack and A > stack[-1][0] and C < stack[-1][1]:
        stack.pop()
    stack.append((A, C))

result = [i + 1 for i, (A, C) in enumerate(stack)]
print(len(result))
print(" ".join(map(str, result)))