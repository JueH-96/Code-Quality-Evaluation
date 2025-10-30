# YOUR CODE HERE
n, x = map(int, input().split())
s = list(map(int, input().split()))

total_score = 0
for score in s:
    if score <= x:
        total_score += score

print(total_score)