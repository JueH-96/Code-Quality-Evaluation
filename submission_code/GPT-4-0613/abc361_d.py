N = int(input().strip())
S = list(input().strip())
T = list(input().strip())

def solve(S, T):
    S = S[::-1]
    T = T[::-1]
    j = 0
    ans = 0
    for i in range(N):
        if S[i] == 'B':
            while j < N and T[j] == 'W':
                j += 1
            if j == N or i > j:
                return -1
            ans += j - i
            j += 1
    return ans

print(solve(S, T))