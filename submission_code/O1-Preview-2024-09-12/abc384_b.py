# YOUR CODE HERE
N, R = map(int, input().split())
rating = R
for _ in range(N):
    D_i, A_i = map(int, input().split())
    T = rating  # Rating at the start of the contest
    if D_i == 1:
        # Division 1
        if 1600 <= T <= 2799:
            rating += A_i
    elif D_i == 2:
        # Division 2
        if 1200 <= T <= 2399:
            rating += A_i
# Print the final rating after N contests
print(rating)