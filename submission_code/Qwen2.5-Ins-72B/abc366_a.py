# YOUR CODE HERE
N, T, A = map(int, input().split())

remaining_votes = N - (T + A)

if T + remaining_votes > A or A + remaining_votes > T:
    print("No")
else:
    print("Yes")