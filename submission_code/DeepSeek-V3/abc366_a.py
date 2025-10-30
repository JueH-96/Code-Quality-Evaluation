# YOUR CODE HERE
N, T, A = map(int, input().split())
remaining = N - (T + A)
max_possible_A = A + remaining
if T > max_possible_A:
    print("Yes")
else:
    print("No")