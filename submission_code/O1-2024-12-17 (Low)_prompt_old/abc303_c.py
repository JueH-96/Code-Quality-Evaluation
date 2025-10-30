def solve():
    import sys

    data = sys.stdin.read().strip().split()
    N, M, H, K = map(int, data[:4])
    S = data[4]
    coords = data[5:]
    
    # Read item coordinates
    items = set()
    idx = 0
    for _ in range(M):
        x = int(coords[idx])
        y = int(coords[idx+1])
        items.add((x, y))
        idx += 2

    # Initialize position and health
    x, y = 0, 0
    health = H

    # Process each move
    for c in S:
        # Move
        if c == 'R':
            x += 1
        elif c == 'L':
            x -= 1
        elif c == 'U':
            y += 1
        elif c == 'D':
            y -= 1

        # Decrease health by 1
        health -= 1
        
        # Check if collapsed
        if health < 0:
            print("No")
            return

        # If there's an item here and health < K, consume it
        if (x, y) in items and health < K:
            health = K
            items.remove((x, y))

    # If finished all moves without collapsing
    print("Yes")

def main():
    solve()

if __name__ == "__main__":
    main()