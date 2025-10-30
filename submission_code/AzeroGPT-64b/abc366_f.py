def mul(A, B):
    return [A[0] * B[0] + A[1] * B[2], A[0] * B[1] + A[1] * B[3], A[2] * B[2] + A[3] * B[2], A[2] * B[1] + A[3] * B[3]]

def pow(A, n, memo):
    if n in memo:
        return memo[n]
    if n == 1:
        memo[n] = A.copy()
        return memo[n]
    if n % 2 == 0:
        memo[n] = pow(mul(A, A), n // 2, memo)
    else:
        memo[n] = mul(A, pow(A, n - 1, memo))
    return memo[n]

n, k = map(int, input().split())
a, b = [0] * n, [0] * n
for i in range(n):
    a[i], b[i] = map(int, input().split())

ans = 0

basis = [i for i in range(n) if a[i] != 1]

combination = 0
for i in range(k):
    if i >= len(basis):
        break
    combination += 1
    A = [a[basis[i]], b[basis[i]], 0, 1]
    for _ in range(k - i - 1):
        A = pow(A, max(1, combination - 1), {})
    ans = max(ans, mul(A, [1, b[basis[i]], 1, 1])[1])
    combination += 1

base = [0] * max(0, n - len(basis))
for i in range(n):
    if a[i] == 1:
        base[i] = b[i]

for i in range(n - len(basis)):
    for j in range(n):
        if a[j] != 1:
            continue
        ans = max(ans, a[basis[-1]] ** (k - 1) * base[i] + a[basis[-1]] ** (k - 2) * b[basis[-1]] + sum(base) * (1 - a[basis[-1]] ** (k - 1)) // (1 - a[basis[-1]]) + b[j])

print(ans)