# YOUR CODE HERE
import sys

def process_balls(N, A):
    stack = []
    for i in range(N):
        size = 2 ** A[i]
        stack.append(size)
        while len(stack) > 1 and stack[-1] == stack[-2]:
            stack.pop()
            stack.pop()
            stack.append(2 * size)
    return len(stack)

N = int(input())
A = list(map(int, input().split()))

print(process_balls(N, A))