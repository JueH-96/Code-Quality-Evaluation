import sys
import threading

def main():
    import sys

    data = sys.stdin.read().split()
    it = iter(data)
    N = int(next(it))
    M = int(next(it))
    H = int(next(it))
    K = int(next(it))
    S = next(it)
    items = set()
    for _ in range(M):
        x = int(next(it)); y = int(next(it))
        items.add((x, y))

    x = 0
    y = 0
    health = H

    for c in S:
        # consume 1 health for the move
        health -= 1
        if health < 0:
            print("No")
            return
        # move
        if c == 'R':
            x += 1
        elif c == 'L':
            x -= 1
        elif c == 'U':
            y += 1
        else:  # 'D'
            y -= 1

        # check for item
        if health < K:
            if (x, y) in items:
                health = K
                items.remove((x, y))

    # if we complete all moves alive
    print("Yes")

if __name__ == "__main__":
    main()