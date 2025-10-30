import sys
import threading

def main():
    import sys

    data = sys.stdin.read().split()
    N = int(data[0])
    A = list(map(int, data[1:]))

    # Build the "next" pointers: for each person j, who stands right behind j?
    nxt = [0] * (N + 1)
    head = -1

    for i in range(1, N + 1):
        pred = A[i - 1]
        if pred == -1:
            head = i
        else:
            nxt[pred] = i

    # Traverse from head following nxt pointers
    res = []
    cur = head
    while cur != 0:
        res.append(str(cur))
        cur = nxt[cur]

    sys.stdout.write(" ".join(res))

if __name__ == "__main__":
    main()