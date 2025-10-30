# YOUR CODE HERE
N, X = map(int, input().split())
scores = list(map(int, input().split()))

total = 0
for score in scores:
    if score <= X:
        total += score

print(total)