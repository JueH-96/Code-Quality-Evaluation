# YOUR CODE HERE
N, L = map(int, input().split())
scores = list(map(int, input().split()))

count = 0
for score in scores:
    if score >= L:
        count += 1

print(count)