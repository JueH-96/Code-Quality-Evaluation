import sys
import threading

def main():
    import sys
    input = sys.stdin.readline

    N = int(input())
    A = list(map(int, input().split()))
    Q = int(input())

    # We maintain a doubly linked list via dictionaries:
    # nxt[v] = the value immediately after v (or None if v is tail)
    # prv[v] = the value immediately before v (or None if v is head)
    nxt = {}
    prv = {}

    # Build the initial linked list
    for i in range(N):
        v = A[i]
        if i == 0:
            prv[v] = None
        else:
            prv[v] = A[i-1]
        if i == N-1:
            nxt[v] = None
        else:
            nxt[v] = A[i+1]

    head = A[0]  # track the head of the list

    # Process queries
    for _ in range(Q):
        q = input().split()
        t = int(q[0])
        if t == 1:
            # insert y after x
            x = int(q[1])
            y = int(q[2])
            # x exists, y is new
            old_n = nxt[x]
            nxt[x] = y
            prv[y] = x
            nxt[y] = old_n
            if old_n is not None:
                prv[old_n] = y
        else:
            # remove x
            x = int(q[1])
            p = prv[x]
            n = nxt[x]
            if p is not None:
                nxt[p] = n
            else:
                # x was head
                head = n
            if n is not None:
                prv[n] = p
            # Optionally delete x from dicts to save memory
            # but not strictly necessary for correctness
            del prv[x]
            del nxt[x]

    # Traverse from head to build the result
    res = []
    cur = head
    while cur is not None:
        res.append(cur)
        cur = nxt[cur]

    # Output
    print(" ".join(map(str, res)))

if __name__ == "__main__":
    main()