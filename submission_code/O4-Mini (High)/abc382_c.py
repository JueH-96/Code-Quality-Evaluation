import sys

def main():
    input = sys.stdin.readline
    N, M = map(int, input().split())
    A = list(map(int, input().split()))
    B = list(map(int, input().split()))

    # We'll build an array minPos[v] = smallest index i such that A[i] == v (or INF if none),
    # then build best[x] = min_{v <= x} minPos[v], which gives the first person with A[i] <= x.
    maxA = max(A)
    maxB = max(B)
    maxV = max(maxA, maxB)
    INF = N + 1

    # Initialize minPos to INF
    minPos = [INF] * (maxV + 1)
    # Record the minimal position for each gourmet level
    for i, v in enumerate(A, start=1):
        if i < minPos[v]:
            minPos[v] = i

    # Build prefix minima: best[x] = earliest person index with A[i] <= x
    best = [INF] * (maxV + 1)
    for x in range(1, maxV + 1):
        # take the smaller of best[x-1] and minPos[x]
        if best[x-1] <= minPos[x]:
            best[x] = best[x-1]
        else:
            best[x] = minPos[x]

    # Answer queries
    out = []
    for x in B:
        ans = best[x]
        if ans > N:
            out.append("-1")
        else:
            out.append(str(ans))

    sys.stdout.write("
".join(out))

# Call main as required
main()