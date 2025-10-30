import sys
import threading
def main():
    import sys
    from collections import deque
    sys.setrecursionlimit(10**7)
    data = sys.stdin
    M = 998244353

    line = data.readline().split()
    if not line:
        return
    H = int(line[0]); W = int(line[1])
    n = H * W

    # label flat array:
    # for green '#' => 0 (unlabeled)
    # for red   '.' => -1
    label = [0] * (n)
    idx = 0
    for _ in range(H):
        row = data.readline().rstrip()
        for c in row:
            if c == '#':
                label[idx] = 0
            else:
                label[idx] = -1
            idx += 1

    # BFS to label green components
    comp_id = 0
    dq = deque()
    for u in range(n):
        if label[u] == 0:
            comp_id += 1
            label[u] = comp_id
            dq.append(u)
            while dq:
                v = dq.popleft()
                # decode row, col
                r = v // W; c = v - r * W
                # up
                if r > 0:
                    v2 = v - W
                    if label[v2] == 0:
                        label[v2] = comp_id
                        dq.append(v2)
                # down
                if r + 1 < H:
                    v2 = v + W
                    if label[v2] == 0:
                        label[v2] = comp_id
                        dq.append(v2)
                # left
                if c > 0:
                    v2 = v - 1
                    if label[v2] == 0:
                        label[v2] = comp_id
                        dq.append(v2)
                # right
                if c + 1 < W:
                    v2 = v + 1
                    if label[v2] == 0:
                        label[v2] = comp_id
                        dq.append(v2)

    C0 = comp_id

    # Count red cells N and sum of adjacent distinct component counts
    N = 0
    sum_adj = 0
    for u in range(n):
        if label[u] == -1:
            N += 1
            # check neighbors for green component ids
            r = u // W; c = u - r * W
            # collect up to 4 neighbor comp ids
            # small list to dedupe
            neigh = []
            if r > 0:
                v2 = u - W
                cid = label[v2]
                if cid > 0:
                    neigh.append(cid)
            if r + 1 < H:
                v2 = u + W
                cid = label[v2]
                if cid > 0:
                    neigh.append(cid)
            if c > 0:
                v2 = u - 1
                cid = label[v2]
                if cid > 0:
                    neigh.append(cid)
            if c + 1 < W:
                v2 = u + 1
                cid = label[v2]
                if cid > 0:
                    neigh.append(cid)
            if neigh:
                # count distinct ids in neigh
                # small list, use set
                sum_adj += len(set(neigh))
            # if no neighbors, adds 0

    # P = (C0+1)*N - sum_adj, Q = N
    # result = P/Q mod M
    P = (C0 + 1) * N - sum_adj
    P %= M
    invN = pow(N, M-2, M)
    ans = P * invN % M
    sys.stdout.write(str(ans))

if __name__ == "__main__":
    main()