# YOUR CODE HERE
N = int(input())
A = list(map(int, input().split()))

line = []
for i in range(N):
    if A[i] == -1:
        line.insert(0, i+1)
    else:
        line.insert(line.index(A[i])+1, i+1)

print(*line)