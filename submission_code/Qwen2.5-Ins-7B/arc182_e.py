# N M C K
N, M, C, K = map(int, input().split())
# A_1 A_2 ... A_N
A = list(map(int, input().split()))

# Calculate the minimum value for the first period
min_values = [min((C * k + a) % M for a in A) for k in range(K)]

# Calculate the sum of the minimum values
result = sum(min_values)

print(result)