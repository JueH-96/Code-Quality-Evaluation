# Read input
n = int(input())
a = list(map(int, input().split()))
b = list(map(int, input().split()))

# Find maximum elements
max_a = max(a)
max_b = max(b)

# Output the sum
print(max_a + max_b)