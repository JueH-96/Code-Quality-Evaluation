# Read input values
N = int(input())
A = list(map(int, input().split()))

# Calculate the sum of the given scores
sum_A = sum(A)

# Determine the score of the Nth person
score_N = -sum_A

# Print the result
print(score_N)