A, M, L, R = map(int, input().split())

lower_bound_k = ((L - A) + M - 1) // M
upper_bound_k = (R - A) // M

result = max(0, upper_bound_k - lower_bound_k + 1)
print(result)