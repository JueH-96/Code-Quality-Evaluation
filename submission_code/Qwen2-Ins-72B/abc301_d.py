S = input()
N = int(input())

def solve(S, N):
    if len(S) == 0:
        return -1
    if S[0] == '?' or S[0] == '1':
        S = '1' + S[1:]
        res = int(S, 2)
        if res <= N:
            return res
        S = '0' + S[1:]
        res = int(S, 2)
        if res <= N:
            return res
        return solve(S[1:], N)
    else:
        return solve(S[1:], N)

print(solve(S, N))