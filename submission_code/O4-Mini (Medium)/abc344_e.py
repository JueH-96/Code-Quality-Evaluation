import sys
import threading

def main():
    import sys

    data = sys.stdin.buffer.read().split()
    it = iter(data)
    # Read initial N and array A
    N = int(next(it))
    A = [int(next(it)) for _ in range(N)]
    # Build doubly linked list via dicts
    nxt = {}
    prv = {}
    for i in range(N):
        x = A[i]
        if i > 0:
            prv[x] = A[i-1]
        else:
            prv[x] = None
        if i < N-1:
            nxt[x] = A[i+1]
        else:
            nxt[x] = None
    head = A[0]

    # Process queries
    Q = int(next(it))
    for _ in range(Q):
        typ = next(it)
        if typ == b'1':
            # insert: 1 x y => insert y after x
            x = int(next(it))
            y = int(next(it))
            # current next of x
            nx = nxt[x]
            # link x -> y
            nxt[x] = y
            prv[y] = x
            # link y -> nx
            nxt[y] = nx
            if nx is not None:
                prv[nx] = y
        else:
            # delete: 2 x
            # remove x from list
            # x guaranteed in list
            x = int(next(it))
            px = prv[x]
            nx = nxt[x]
            if px is not None:
                nxt[px] = nx
            else:
                # x was head
                head = nx
            if nx is not None:
                prv[nx] = px
            # optional: remove x from dicts, but not needed

    # Traverse from head and collect result
    res = []
    cur = head
    while cur is not None:
        res.append(str(cur))
        cur = nxt[cur]
    sys.stdout.write(" ".join(res))

if __name__ == "__main__":
    main()