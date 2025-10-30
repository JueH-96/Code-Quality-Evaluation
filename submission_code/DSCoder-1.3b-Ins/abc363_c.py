import sys

def solve():
    N, K = map(int, sys.stdin.readline().split())
    S = sys.stdin.readline().strip()

    def check(s):
        for i in range(len(s) - K + 1):
            if s[i:i+K] == s[i:i+K][::-1]:
                return False
        return True

    count = 0
    for i in range(N):
        for j in range(i+1, N+1):
            if check(S[i:j]):
                count += 1

    print(count)

solve()