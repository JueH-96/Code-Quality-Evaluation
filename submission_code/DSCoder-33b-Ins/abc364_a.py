# YOUR CODE HERE

N = int(input())
S = [input() for _ in range(N)]

for i in range(N-1):
    if S[i] == 'sweet' and S[i+1] == 'sweet':
        print('No')
        break
else:
    print('Yes')