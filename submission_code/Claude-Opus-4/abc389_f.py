# YOUR CODE HERE
N = int(input())
contests = []
for _ in range(N):
    L, R = map(int, input().split())
    contests.append((L, R))

Q = int(input())

# For each possible rating value, compute how much it increases
max_rating = 500000
rating_increase = [0] * (max_rating + 1)

for rating in range(1, max_rating + 1):
    current = rating
    for L, R in contests:
        if L <= current <= R:
            current += 1
    rating_increase[rating] = current - rating

# Answer queries
for _ in range(Q):
    X = int(input())
    print(X + rating_increase[X])