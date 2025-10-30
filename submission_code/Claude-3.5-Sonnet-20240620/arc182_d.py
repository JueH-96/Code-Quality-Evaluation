# YOUR CODE HERE
def min_operations(N, M, A, B):
    def distance(a, b):
        return min((b - a) % M, (a - b) % M)

    total_ops = 0
    for i in range(N):
        if A[i] != B[i]:
            if i == 0:
                total_ops += distance(A[i], B[i])
            elif i == N - 1:
                total_ops += distance(A[i], B[i])
            else:
                if A[i-1] != B[i] and A[i+1] != B[i]:
                    total_ops += distance(A[i], B[i])
                else:
                    return -1
    return total_ops

N, M = map(int, input().split())
A = list(map(int, input().split()))
B = list(map(int, input().split()))

print(min_operations(N, M, A, B))