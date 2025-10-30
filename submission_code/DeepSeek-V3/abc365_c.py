def find_max_x(N, M, A):
    A_sorted = sorted(A)
    total = 0
    for i in range(N):
        if total + A_sorted[i] * (N - i) <= M:
            total += A_sorted[i]
        else:
            x = (M - total) // (N - i)
            return x
    return "infinite"

# Read input
N, M = map(int, input().split())
A = list(map(int, input().split()))

# Compute and print the result
result = find_max_x(N, M, A)
print(result)