# YOUR CODE HERE

N = int(input())
A = [list(map(int, input().split())) for _ in range(N)]

element = 1
for i in range(N):
    element = A[element-1][i]

print(element)