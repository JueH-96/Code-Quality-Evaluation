# YOUR CODE HERE
N, T, A = map(int, input().split())

if T + A > N or T + A < 0 or T > N - A or T < 0:
    print("No")
else:
    print("Yes")