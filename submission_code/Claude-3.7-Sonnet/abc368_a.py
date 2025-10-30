# Read input
N, K = map(int, input().split())
A = list(map(int, input().split()))

# Take K cards from the bottom and place them on top
# The last K cards go to the top while maintaining their order
B = A[N-K:] + A[:N-K]

# Print the result
print(*B)