N, M, P = map(int, input().split())
A = list(map(int, input().split()))
B = list(map(int, input().split()))

min_prices = [min(A[i] + B[j], P) for i in range(N) for j in range(M)]
print(sum(min_prices))