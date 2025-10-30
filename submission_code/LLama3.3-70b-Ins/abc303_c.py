import sys

def main():
    # Read input
    N, M, H, K = map(int, sys.stdin.readline().split())
    S = sys.stdin.readline().strip()
    items = []
    for _ in range(M):
        x, y = map(int, sys.stdin.readline().split())
        items.append((x, y))

    # Initialize variables
    x, y = 0, 0
    health = H

    # Simulate moves
    for move in S:
        # Move
        if move == 'R':
            x += 1
        elif move == 'L':
            x -= 1
        elif move == 'U':
            y += 1
        elif move == 'D':
            y -= 1

        # Consume health
        health -= 1

        # Check if health is negative
        if health < 0:
            print("No")
            return

        # Check if item is available
        if (x, y) in items and health < K:
            health = K
            items.remove((x, y))

    print("Yes")

if __name__ == "__main__":
    main()