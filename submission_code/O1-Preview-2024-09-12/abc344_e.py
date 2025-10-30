# YOUR CODE HERE
import sys
import threading

def main():
    import sys

    sys.setrecursionlimit(1 << 25)
    N = int(sys.stdin.readline())
    A_list = list(map(int, sys.stdin.readline().split()))
    Q = int(sys.stdin.readline())

    prev = {}
    next = {}

    head = A_list[0]
    for i, x in enumerate(A_list):
        if i == 0:
            prev[x] = None
        else:
            prev[x] = A_list[i - 1]
        if i == len(A_list) -1:
            next[x] = None
        else:
            next[x] = A_list[i +1]

    for _ in range(Q):
        query = sys.stdin.readline().split()
        if query[0] == '1':
            x = int(query[1])
            y = int(query[2])
            # Insert y immediately after x
            prev[y] = x
            next[y] = next.get(x)
            next[x] = y
            if next[y] is not None:
                prev[next[y]] = y
        else:
            x = int(query[1])
            # Remove x
            prev_x = prev.get(x)
            next_x = next.get(x)
            if prev_x is None:
                # x is head
                head = next_x
            else:
                next[prev_x] = next_x
            if next_x is not None:
                prev[next_x] = prev_x
            # Optionally delete x from prev and next dictionaries
            # del prev[x]
            # del next[x]

    # Output the result
    res = []
    current = head
    while current is not None:
        res.append(str(current))
        current = next.get(current)
    print(' '.join(res))


if __name__ == "__main__":
    threading.Thread(target=main).start()