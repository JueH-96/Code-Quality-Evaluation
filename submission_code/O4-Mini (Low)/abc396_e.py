import sys
import threading
def main():
    import sys
    sys.setrecursionlimit(10**7)
    data = sys.stdin
    line = data.readline().split()
    if not line:
        print(-1)
        return
    N, M = map(int, line)
    adj = [[] for _ in range(N)]
    for _ in range(M):
        x, y, z = map(int, data.readline().split())
        x -= 1
        y -= 1
        # constraint A[x] XOR A[y] = z
        adj[x].append((y, z))
        adj[y].append((x, z))
    from collections import deque
    label = [None]*N   # assigned XOR-distance from component root
    ans = [0]*N
    for i in range(N):
        if label[i] is not None:
            continue
        # BFS this component
        label[i] = 0
        comp = [i]
        dq = deque([i])
        ok = True
        while dq and ok:
            u = dq.popleft()
            Lu = label[u]
            for v, w in adj[u]:
                if label[v] is None:
                    # L[v] = L[u] XOR w
                    label[v] = Lu ^ w
                    comp.append(v)
                    dq.append(v)
                else:
                    # check consistency
                    if (Lu ^ label[v]) != w:
                        ok = False
                        break
        if not ok:
            print(-1)
            return
        # Now we have labels L for this component. We may choose a T to XOR all
        # so as to minimize sum of (L[v] ^ T).
        # We'll pick T bit by bit (up to 31 bits).
        size = len(comp)
        # count ones per bit
        bitcnt = [0]*31
        for v in comp:
            Lv = label[v]
            for b in range(31):
                if (Lv >> b) & 1:
                    bitcnt[b] += 1
        T = 0
        # for each bit choose T_b = 0 or 1
        for b in range(31):
            cnt1 = bitcnt[b]
            cnt0 = size - cnt1
            # if T_b=0 cost = cnt1 * 2^b; if T_b=1 cost = cnt0 * 2^b
            if cnt0 < cnt1:
                # pick T_b = 1
                T |= (1 << b)
        # assign answers
        for v in comp:
            ans[v] = label[v] ^ T
    # finally print ans
    out = sys.stdout
    out.write(" ".join(str(x) for x in ans))

if __name__ == "__main__":
    main()