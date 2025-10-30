# N = int(input())
# A = list(map(int, input().split()))

def min_operations(N, A):
    A.sort()
    min_val = A[0]
    max_val = A[-1]
    operations = 0
    for i in range(1, N):
        diff = max_val - min_val
        if diff <= 1:
            break
        operations += (A[i] - min_val - 1)
        min_val += 1
        max_val = A[i]
    return operations

N = int(input())
A = list(map(int, input().split()))
print(min_operations(N, A))