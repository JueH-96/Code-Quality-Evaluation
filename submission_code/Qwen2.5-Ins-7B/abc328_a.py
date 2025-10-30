# YOUR CODE HERE
n, x = map(int, input().split())
scores = list(map(int, input().split()))
total_score = sum(score for score in scores if score <= x)
print(total_score)