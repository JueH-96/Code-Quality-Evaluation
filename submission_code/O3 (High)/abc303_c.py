import sys


def main() -> None:
    sys.setrecursionlimit(1 << 25)
    N, M, H, K = map(int, sys.stdin.readline().split())
    S = sys.stdin.readline().strip()

    # store positions of the items
    items = set()
    for _ in range(M):
        x, y = map(int, sys.stdin.readline().split())
        items.add((x, y))

    x = y = 0               # current position
    health = H              # current health

    for c in S:
        # move
        if c == 'R':
            x += 1
        elif c == 'L':
            x -= 1
        elif c == 'U':
            y += 1
        else:               # 'D'
            y -= 1

        # spend one health to move
        health -= 1

        # if health is negative, he collapses
        if health < 0:
            print("No")
            return

        # consume an item if present and health < K
        pos = (x, y)
        if pos in items and health < K:
            health = K
            items.remove(pos)

    # finished all moves successfully
    print("Yes")


if __name__ == "__main__":
    main()