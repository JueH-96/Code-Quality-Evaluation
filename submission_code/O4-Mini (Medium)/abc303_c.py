import sys
import threading

def main():
    import sys

    input = sys.stdin.readline
    N, M, H, K = map(int, input().split())
    S = input().strip()

    # Read item coordinates into a set for O(1) lookups/removals
    items = set()
    for _ in range(M):
        x, y = map(int, input().split())
        items.add((x, y))

    x = 0
    y = 0
    health = H

    for c in S:
        # Move according to c
        if c == 'R':
            x += 1
        elif c == 'L':
            x -= 1
        elif c == 'U':
            y += 1
        elif c == 'D':
            y -= 1

        # Consume 1 health for the move
        health -= 1

        # If health is negative, he collapses
        if health < 0:
            print("No")
            return

        # If there's an item here and health < K, consume it to reset health to K
        if (x, y) in items and health < K:
            health = K
            items.remove((x, y))

    # If we finish all moves without health < 0, it's a success
    print("Yes")

if __name__ == "__main__":
    main()