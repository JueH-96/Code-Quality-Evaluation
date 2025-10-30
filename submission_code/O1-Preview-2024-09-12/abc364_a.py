# YOUR CODE HERE
N = int(input())
S = [input().strip() for _ in range(N)]

for i in range(1, N):
    if S[i - 1] == 'sweet' and S[i] == 'sweet':
        if i == N - 1:
            break
        else:
            print('No')
            exit()
print('Yes')