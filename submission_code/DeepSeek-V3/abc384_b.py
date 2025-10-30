# YOUR CODE HERE
N, R = map(int, input().split())
contests = []
for _ in range(N):
    D, A = map(int, input().split())
    contests.append((D, A))

current_rating = R
for D, A in contests:
    if D == 1:
        if 1600 <= current_rating <= 2799:
            current_rating += A
    else:
        if 1200 <= current_rating <= 2399:
            current_rating += A

print(current_rating)