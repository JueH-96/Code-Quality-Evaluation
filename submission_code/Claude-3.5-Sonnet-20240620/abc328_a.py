# YOUR CODE HERE
N, X = map(int, input().split())
scores = list(map(int, input().split()))

total_score = sum(score for score in scores if score <= X)
print(total_score)