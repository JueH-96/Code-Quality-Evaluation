# YOUR CODE HERE
n = int(input())
scores = list(map(int, input().split()))
max_score = max(scores[1:])
needed_points = max(0, max_score - scores[0] + 1)
print(needed_points)