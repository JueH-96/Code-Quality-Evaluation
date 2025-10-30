# YOUR CODE HERE
N = int(input())
S = input().split()

sick = False
for i in range(N):
    if S[i] == 'salty' and S[i] == S[i+1]:
        sick = True
        break

if sick:
    print('No')
else:
    print('Yes')