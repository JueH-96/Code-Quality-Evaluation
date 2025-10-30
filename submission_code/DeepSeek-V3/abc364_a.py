# YOUR CODE HERE
N = int(input())
S = [input().strip() for _ in range(N)]
for i in range(1, N):
    if S[i] == 'sweet' and S[i-1] == 'sweet':
        print("No")
        exit()
print("Yes")