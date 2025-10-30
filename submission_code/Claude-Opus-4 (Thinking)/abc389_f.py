# YOUR CODE HERE
n = int(input())
contests = []
for _ in range(n):
    l, r = map(int, input().split())
    contests.append((l, r))

q = int(input())
for _ in range(q):
    x = int(input())
    rating = x
    for l, r in contests:
        if l <= rating <= r:
            rating += 1
    print(rating)