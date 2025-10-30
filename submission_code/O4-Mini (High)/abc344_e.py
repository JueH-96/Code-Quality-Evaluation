import sys
def main():
    input = sys.stdin.readline
    N = int(input())
    A = list(map(int, input().split()))
    # We use two dictionaries to simulate a doubly-linked list:
    # prev[x] = the element immediately before x (0 if x is head)
    # nxt[x]  = the element immediately after x (0 if x is tail)
    prev = {}
    nxt = {}

    # Initialize the linked list with the initial sequence A
    head = A[0]
    prev[head] = 0
    for i in range(N-1):
        nxt[A[i]] = A[i+1]
        prev[A[i+1]] = A[i]
    nxt[A[-1]] = 0

    Q = int(input())
    for _ in range(Q):
        q = input().split()
        if q[0] == '1':
            # Query "1 x y": insert y immediately after x
            x = int(q[1])
            y = int(q[2])
            nx = nxt[x]      # the old next of x
            # link x -> y -> nx
            nxt[x] = y
            prev[y] = x
            nxt[y] = nx
            if nx != 0:
                prev[nx] = y
        else:
            # Query "2 x": remove x
            x = int(q[1])
            px = prev.get(x, 0)
            nx = nxt.get(x, 0)
            if px != 0:
                nxt[px] = nx
            else:
                # x was head
                head = nx
            if nx != 0:
                prev[nx] = px

    # Traverse from head to tail and collect the result
    res = []
    cur = head
    while cur != 0:
        res.append(str(cur))
        cur = nxt.get(cur, 0)

    print(' '.join(res))


if __name__ == "__main__":
    main()