# YOUR CODE HERE
N = int(input())
S = input()

for n in range(1, N - 1):
    if S[n-1:n+3] == "ABC":
        print(n)
        break
else:
    print(-1)