# YOUR CODE HERE
N = int(input())
scores = list(map(int, input().split()))

# Calculate the sum of all scores
total_score = sum(scores)

# The sum of all scores (including person N's score) must be zero
# because for each game, one person gains a point and another loses a point
# So, person N's score is the negative of the sum of all other scores
n_score = -total_score

print(n_score)