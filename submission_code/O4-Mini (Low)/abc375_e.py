import sys
import threading

def main():
    import sys
    input = sys.stdin.readline

    N = int(input().strip())
    A = [0]*N
    B = [0]*N
    total = 0
    for i in range(N):
        a, b = map(int, input().split())
        A[i] = a
        B[i] = b
        total += b

    # total strength must be divisible by 3
    if total % 3 != 0:
        print(-1)
        return
    T = total // 3

    # dp[a][b] = max number of people who stay in original team
    # when team1 sum is a, team2 sum is b (team3 sum implicit)
    # initialize
    # use -1 for impossible
    dp = [[-1]*(T+1) for _ in range(T+1)]
    dp[0][0] = 0

    for i in range(N):
        orig = A[i]
        w = B[i]
        # next dp
        ndp = [[-1]*(T+1) for _ in range(T+1)]
        # iterate old states
        # local references
        dpi = dp
        ndpi = ndp
        T1 = T
        o1 = 1 if orig == 1 else 0
        o2 = 1 if orig == 2 else 0
        o3 = 1 if orig == 3 else 0
        for a in range(T1+1):
            row = dpi[a]
            for b in range(T1+1):
                v = row[b]
                if v < 0:
                    continue
                # assign to team1
                na = a + w
                if na <= T1:
                    vb = v + o1
                    if vb > ndpi[na][b]:
                        ndpi[na][b] = vb
                # assign to team2
                nb = b + w
                if nb <= T1:
                    vb = v + o2
                    if vb > ndpi[a][nb]:
                        ndpi[a][nb] = vb
                # assign to team3
                # a,b unchanged
                vb = v + o3
                if vb > ndpi[a][b]:
                    ndpi[a][b] = vb
        dp = ndp

    # answer at dp[T][T]
    best_stay = dp[T][T]
    if best_stay < 0:
        print(-1)
    else:
        # moves = total people - stays
        print(N - best_stay)

if __name__ == "__main__":
    main()