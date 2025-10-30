import sys
# read inputs
N = int(input())
S = input()

# solve
for i in range(N-2):
    if S[i:i+3] == 'ABC':
        print(i+1)
        exit()
print(-1)