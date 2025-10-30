# YOUR CODE HERE
N, X = map(int, input().split())
S = list(map(int, input().split()))

total_score = 0
for score in S:
    if score <= X:
        total_score += score

print(total_score)