import sys

N, M, L = map(int, sys.stdin.readline().split())
A = list(map(int, sys.stdin.readline().split()))
B = list(map(int, sys.stdin.readline().split()))
CD = [list(map(int, sys.stdin.readline().split())) for _ in range(L)]

# Create a list to store the prices of the meals that are not disliked
prices = []

# Add the prices of the meals that are not disliked to the list
for i in range(1, N+1):
    for j in range(1, M+1):
        if [i, j] not in CD:
            prices.append(A[i-1] + B[j-1])

# Print the maximum price
print(max(prices))