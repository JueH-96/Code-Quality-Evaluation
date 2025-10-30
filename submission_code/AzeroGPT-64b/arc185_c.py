import sys

def solve(A, n, X):
    if n < 3:
        return [-1]

    A.sort()

    for i in range(n - 2):
        a = A[i]
        if a * 3 > X:
            return [-1]

        j, k = i + 1, n - 1

        while j < k:
            b, c = A[j], A[k]

            if a + b + c == X:
                return [i + 1, j + 1, k + 1]
            elif a + b + c < X:
                j += 1
            else:
                k -= 1

    return [-1]

# Read from stdin
n, X = map(int, sys.stdin.readline().split())
A = list(map(int, sys.stdin.readline().split()))

# Calculate result
result = solve(A, n, X)

# Output to stdout
print(*result)