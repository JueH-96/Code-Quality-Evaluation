def count_arithmetic_subsequences(N, A):
    if N == 0:
        return 0
    total = 0
    l = 0
    while l < N:
        r = l
        if r + 1 < N:
            d = A[r+1] - A[r]
            r += 1
            while r + 1 < N and A[r+1] - A[r] == d:
                r += 1
        else:
            r = l
        length = r - l + 1
        total += length * (length + 1) // 2
        l = r + 1
    return total

# Read input
N = int(input())
A = list(map(int, input().split()))

# Compute and print the result
print(count_arithmetic_subsequences(N, A))