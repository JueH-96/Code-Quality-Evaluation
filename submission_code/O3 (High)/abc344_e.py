import sys

def main() -> None:
    data = list(map(int, sys.stdin.buffer.read().split()))
    it = 0

    # initial sequence
    N = data[it]; it += 1
    A = data[it: it + N]; it += N

    # linked-list representation
    prev = {}
    nxt  = {}
    for i, v in enumerate(A):
        prev[v] = A[i - 1] if i else None
        nxt[v]  = A[i + 1] if i + 1 < N else None
    head = A[0]

    # queries
    Q = data[it]; it += 1
    for _ in range(Q):
        typ = data[it]; it += 1

        if typ == 1:                      # insert
            x = data[it]; y = data[it + 1]; it += 2

            nx = nxt.get(x)               # node currently after x
            prev[y] = x
            nxt[y]  = nx
            nxt[x]  = y
            if nx is not None:
                prev[nx] = y

        else:                             # delete
            x = data[it]; it += 1

            pv = prev.get(x)
            nx = nxt.get(x)

            if pv is None:
                head = nx                 # x was the head
            else:
                nxt[pv] = nx
            if nx is not None:
                prev[nx] = pv

            # remove x’s own links
            prev.pop(x, None)
            nxt.pop(x,  None)

    # output final sequence
    res = []
    cur = head
    while cur is not None:
        res.append(str(cur))
        cur = nxt.get(cur)
    print(' '.join(res))

if __name__ == "__main__":
    main()