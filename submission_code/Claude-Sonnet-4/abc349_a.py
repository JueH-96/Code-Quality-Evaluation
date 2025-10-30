# YOUR CODE HERE
n = int(input())
scores = list(map(int, input().split()))

# The sum of all scores must be 0
# So person N's score = -(sum of all other scores)
person_n_score = -sum(scores)

print(person_n_score)