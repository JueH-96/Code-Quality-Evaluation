# YOUR CODE HERE
N = int(input())
prev = ''
for _ in range(N):
    S = input()
    if S == 'sweet' and S == prev:
        print('No')
        break
    prev = S
else:
    print('Yes')