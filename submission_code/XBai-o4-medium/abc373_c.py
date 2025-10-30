# Read input
n = int(input())
A = list(map(int, input().split()))
B = list(map(int, input().split()))

# Compute maximum values
max_A = max(A)
max_B = max(B)

# Output the sum
print(max_A + max_B)