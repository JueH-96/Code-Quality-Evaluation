def main():
    import sys
    from itertools import permutations

    input = sys.stdin.readline
    H, W = map(int, input().split())
    A = [list(map(int, input().split())) for _ in range(H)]
    B = [list(map(int, input().split())) for _ in range(H)]

    ans = float('inf')
    rows = range(H)
    cols = range(W)

    for P in permutations(rows):
        # inversion count for row permutation P
        costP = 0
        for i in range(H):
            for j in range(i+1, H):
                if P[i] > P[j]:
                    costP += 1
        if costP >= ans:
            continue

        for Q in permutations(cols):
            # inversion count for column permutation Q
            costQ = 0
            for i in range(W):
                for j in range(i+1, W):
                    if Q[i] > Q[j]:
                        costQ += 1
            totalCost = costP + costQ
            if totalCost >= ans:
                continue

            ok = True
            for i in rows:
                Ai = A[P[i]]
                Bi = B[i]
                for j in cols:
                    if Ai[Q[j]] != Bi[j]:
                        ok = False
                        break
                if not ok:
                    break

            if ok:
                ans = totalCost

    print(-1 if ans == float('inf') else ans)

if __name__ == "__main__":
    main()