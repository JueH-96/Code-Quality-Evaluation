# YOUR CODE HERE
N = int(input())
S = input()
T = input()

for i in range(N):
    if S[i] != T[i] and (S[i] not in 'lo' or T[i] not in 'lo10'):
        print('No')
        break
else:
    print('Yes')