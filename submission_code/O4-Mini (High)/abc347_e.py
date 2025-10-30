import sys
def main():
    data = sys.stdin.read().split()
    if not data:
        return
    it = iter(data)
    N = int(next(it))
    Q = int(next(it))
    # inS[j] = whether j is currently in the set S
    inS = [False] * (N + 1)
    # pre[i] = sum of |S| over queries 1..i
    pre = [0] * (Q + 2)
    # occs[j] = list of query‐indices i where x_i = j
    occs = [[] for _ in range(N + 1)]
    cnt = 0
    for i in range(1, Q + 1):
        x = int(next(it))
        occs[x].append(i)
        if inS[x]:
            cnt -= 1
            inS[x] = False
        else:
            cnt += 1
            inS[x] = True
        pre[i] = pre[i - 1] + cnt

    # Compute final A_j for j=1..N
    ans = [0] * (N + 1)
    for j in range(1, N + 1):
        total = 0
        T = occs[j]
        L = len(T)
        # j is in S in intervals [T[0], T[1]), [T[2], T[3]), ...
        for k in range(0, L, 2):
            l = T[k]
            if k + 1 < L:
                r = T[k + 1]
            else:
                # remains in S until after query Q
                r = Q + 1
            # sum of pre[l..r-1] = pre[r-1] - pre[l-1]
            total += pre[r - 1] - pre[l - 1]
        ans[j] = total

    # output A_1 ... A_N
    sys.stdout.write(' '.join(map(str, ans[1:])))
    
if __name__ == "__main__":
    main()