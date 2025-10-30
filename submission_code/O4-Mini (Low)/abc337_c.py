import sys
import threading

def main():
    import sys

    data = sys.stdin.read().split()
    N = int(data[0])
    A = list(map(int, data[1:]))

    # next_person[i] will store the person standing immediately behind person i
    next_person = [0] * (N + 1)
    head = 0

    for i in range(1, N + 1):
        if A[i-1] == -1:
            head = i
        else:
            next_person[A[i-1]] = i

    # Traverse from the head to the tail, collecting the order
    order = []
    cur = head
    while cur != 0:
        order.append(str(cur))
        cur = next_person[cur]

    sys.stdout.write(" ".join(order))

if __name__ == "__main__":
    main()