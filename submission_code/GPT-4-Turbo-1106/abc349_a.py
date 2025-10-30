# Read the number of people
N = int(input())

# Read the scores of the first N-1 people
scores = list(map(int, input().split()))

# Calculate the score of the Nth person
score_N = -sum(scores)

# Print the score of the Nth person
print(score_N)