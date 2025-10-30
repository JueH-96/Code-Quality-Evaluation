# Read the input
N, K = map(int, input().split())
A = list(map(int, input().split()))

# Create a set of the numbers in A for faster lookup
A_set = set(A)

# Calculate the sum of the numbers between 1 and K that are not in A
total = 0
for i in range(1, K+1):
    if i not in A_set:
        total += i

# Print the answer
print(total)