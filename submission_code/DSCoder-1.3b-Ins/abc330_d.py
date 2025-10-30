import sys

def solve():
    N = int(sys.stdin.readline().strip())
    S = [sys.stdin.readline().strip() for _ in range(N)]

    def count_triples(S):
        count = 0
        for i in range(N):
            for j in range(N):
                if i < j and S[i][j] == 'o':
                    for k in range(j+1, N):
                        if S[i][k] == 'o' and S[i][j] != S[i][k]:
                            count += 1
        return count

    print(count_triples(S))

solve()