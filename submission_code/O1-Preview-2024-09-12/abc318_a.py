# YOUR CODE HERE
N, M, P = map(int, input().split())
if M > N:
    print(0)
else:
    ans = ((N - M) // P) + 1
    print(ans)