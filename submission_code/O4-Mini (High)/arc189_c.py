import sys
import threading
def main():
    import sys
    input = sys.stdin.readline
    N, X = map(int, input().split())
    A = list(map(int, input().split()))
    B = list(map(int, input().split()))
    P_temp = list(map(int, input().split()))
    Q_temp = list(map(int, input().split()))
    # Build P-cycle containing X
    arrP = []
    cur = X
    while True:
        arrP.append(cur)
        nxt = P_temp[cur-1]
        if nxt == X:
            break
        cur = nxt
    Lp = len(arrP)
    idxP = [-1] * (N+1)
    for k, node in enumerate(arrP):
        idxP[node] = k
    # Build Q-cycle containing X
    arrQ = []
    cur = X
    while True:
        arrQ.append(cur)
        nxt = Q_temp[cur-1]
        if nxt == X:
            break
        cur = nxt
    Lq = len(arrQ)
    idxQ = [-1] * (N+1)
    for k, node in enumerate(arrQ):
        idxQ[node] = k
    # Check reachability and compute max distances
    maxDistP = 0
    for i in range(N):
        if A[i] == 1:
            box = i+1
            k = idxP[box]
            if k < 0:
                print(-1)
                return
            # distance along P-edges from box to X
            dist = 0 if k == 0 else (Lp - k)
            if dist > maxDistP:
                maxDistP = dist
    maxDistQ = 0
    for i in range(N):
        if B[i] == 1:
            box = i+1
            k = idxQ[box]
            if k < 0:
                print(-1)
                return
            dist = 0 if k == 0 else (Lq - k)
            if dist > maxDistQ:
                maxDistQ = dist
    # Count how many boxes need an operation
    ans = 0
    for box in range(1, N+1):
        if box == X:
            continue
        need = False
        k = idxP[box]
        if k >= 0:
            distP = 0 if k == 0 else (Lp - k)
            if distP <= maxDistP:
                need = True
        k = idxQ[box]
        if k >= 0:
            distQ = 0 if k == 0 else (Lq - k)
            if distQ <= maxDistQ:
                need = True
        if need:
            ans += 1
    print(ans)

if __name__ == "__main__":
    main()