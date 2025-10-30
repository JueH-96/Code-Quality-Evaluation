N, K, X = map(int, input().split())
A = list(map(int, input().split()))

B = A[:K]
B.append(X)
B.extend(A[K:])

print(*B)