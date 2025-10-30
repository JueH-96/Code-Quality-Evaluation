N = int(input().strip())
A = [[0]*N for _ in range(N)]
for i in range(N):
    A[i][:i+1] = map(int, input().strip().split())

element = 1
for i in range(1, N):
    element = A[element-1][i]
print(element)