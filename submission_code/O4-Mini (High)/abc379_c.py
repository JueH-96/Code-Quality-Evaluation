import sys
def main():
    input = sys.stdin.readline
    N, M = map(int, input().split())
    X = list(map(int, input().split()))
    A = list(map(int, input().split()))
    # total stones must be N
    total = 0
    for a in A:
        total += a
    if total != N:
        print(-1)
        return
    # sort piles by position
    piles = sorted(zip(X, A))
    cum = 0        # cumulative stones seen so far
    prevX = 0     # previous position
    sumAX = 0     # sum of A_i * X_i
    for x, a in piles:
        # between prevX+1 and x-1, cum stones must be >= each j,
        # worst case j = x-1
        if prevX + 1 <= x - 1:
            if cum < x - 1:
                print(-1)
                return
        cum += a
        prevX = x
        sumAX += a * x
    # if we passed all checks, answer is sum of (k - init_pos_k)
    # = (1+...+N) - sum(init_pos_k)
    ans = N * (N + 1) // 2 - sumAX
    print(ans)

main()