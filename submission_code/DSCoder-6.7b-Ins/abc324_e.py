import sys

def solve():
    N, T = sys.stdin.readline().split()
    N = int(N)
    S = [sys.stdin.readline().strip() for _ in range(N)]
    T = T.strip()

    count = 0
    for i in range(1, N+1):
        for j in range(1, N+1):
            if i != j and T in S[i-1]+S[j-1]:
                count += 1
    print(count)

solve()