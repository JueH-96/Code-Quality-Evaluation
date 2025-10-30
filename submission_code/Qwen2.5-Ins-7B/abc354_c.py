# YOUR CODE HERE
from collections import defaultdict

N = int(input())
cards = []
for i in range(N):
    A, C = map(int, input().split())
    cards.append((A, C, i+1))

cards.sort()

stack = []
for A, C, i in cards:
    while stack and stack[-1][0] < A and stack[-1][1] > C:
        stack.pop()
    stack.append((A, C, i))

stack.sort(key=lambda x: x[2])

m = len(stack)
print(m)
for A, C, i in stack:
    print(i, end=' ')
print()