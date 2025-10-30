# YOUR CODE HERE
S = input()
N = int(input())

def solve(S, N):
    S = list(S)
    for i in range(len(S)):
        if S[i] == '?':
            S[i] = '1'
    S = ''.join(S)
    if int(S, 2) > N:
        return -1
    else:
        return int(S, 2)

print(solve(S, N))