# Read the number of integers
N = int(input())

# Read the integers
A = list(map(int, input().split()))

# Find the maximum integer
max_A = max(A)

# Find the largest integer that is not the maximum
second_max_A = max([a for a in A if a != max_A])

# Print the result
print(second_max_A)