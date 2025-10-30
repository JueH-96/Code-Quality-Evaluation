# YOUR CODE HERE
n, x = map(int, input().split())
scores = list(map(int, input().split()))

total = 0
for score in scores:
    if score <= x:
        total += score

print(total)