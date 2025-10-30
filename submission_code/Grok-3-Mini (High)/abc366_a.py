# YOUR CODE HERE
N, T, A = map(int, input().split())
if 2 * T > N or 2 * A > N:
    print("Yes")
else:
    print("No")