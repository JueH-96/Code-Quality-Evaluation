def main():
    import sys
    input_data = sys.stdin.read().split()
    it = iter(input_data)
    N = int(next(it))
    M = int(next(it))
    H = int(next(it))
    K = int(next(it))
    S = next(it)
    
    # Insert items coordinates into a set for O(1) lookup.
    items = set()
    for _ in range(M):
        x = int(next(it))
        y = int(next(it))
        items.add((x, y))
    
    # Start at the origin with initial health H.
    x, y = 0, 0
    health = H

    for move in S:
        # Each move costs 1 health.
        health -= 1
        if health < 0:
            print("No")
            return

        # Update position based on the move.
        if move == 'R':
            x += 1
        elif move == 'L':
            x -= 1
        elif move == 'U':
            y += 1
        elif move == 'D':
            y -= 1

        # If an item exists at the new position and health is less than K, consume it.
        if (x, y) in items and health < K:
            health = K
            items.remove((x, y))
    
    print("Yes")

if __name__ == '__main__':
    main()