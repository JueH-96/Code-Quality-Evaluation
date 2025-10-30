def min_operations(N, A):
    total = sum(A)
    avg = total // N
    remainder = total % N
    target = [avg] * (N - remainder) + [avg + 1] * remainder
    target.sort()
    A.sort()
    operations = 0
    for a, t in zip(A, target):
        operations += abs(a - t)
    return operations // 2

# Read input
N = int(input())
A = list(map(int, input().split()))
# Compute and print the result
print(min_operations(N, A))