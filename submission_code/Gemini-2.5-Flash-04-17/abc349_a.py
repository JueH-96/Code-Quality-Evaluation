# Read N
N = int(input())

# Read the N-1 scores
# Read the line, split by spaces, convert each part to int
# Using list(map(...)) creates a list of the scores
A = list(map(int, input().split()))

# Calculate the sum of the first N-1 scores
# The sum of all N scores must be 0 because each game changes the total score by (+1) + (-1) = 0.
# Initial total score is 0.
# Let A_N be the score of person N.
# The scores given are A_1, A_2, ..., A_{N-1}.
# In the input array A (0-indexed), A[0] is A_1, A[1] is A_2, ..., A[N-2] is A_{N-1}.
# The sum of scores of persons 1 to N-1 is sum(A).
sum_A = sum(A)

# The sum of all N scores is A_1 + A_2 + ... + A_{N-1} + A_N = 0
# Substituting sum_A for the sum of the first N-1 scores:
# sum_A + A_N = 0
# Therefore, the score of person N (A_N) is the negative of the sum of the first N-1 scores.
score_N = -sum_A

# Print the result
print(score_N)