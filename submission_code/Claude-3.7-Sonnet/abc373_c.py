# Read input
N = int(input())
A = list(map(int, input().split()))
B = list(map(int, input().split()))

# Calculate and print the maximum value
print(max(A) + max(B))