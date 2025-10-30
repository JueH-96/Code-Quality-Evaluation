# Read the input
N = int(input())
A = [int(x) for x in input().split()]

# Define the function f(x, y)
def f(x, y):
    return int(str(x) + str(y))

# Calculate the sum of f(A_i, A_j) for all i < j
total = 0
for i in range(N-1):
    for j in range(i+1, N):
        total += f(A[i], A[j])

# Print the answer modulo 998244353
print(total % 998244353)