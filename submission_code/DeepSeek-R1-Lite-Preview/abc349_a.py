# Read the number of people
N = int(input())

# Read the scores of the first N-1 people
A = list(map(int, input().split()))

# Calculate the sum of the scores of the first N-1 people
sum_A = sum(A)

# The score of the Nth person is the negative of this sum
S_N = -sum_A

# Print the result
print(S_N)