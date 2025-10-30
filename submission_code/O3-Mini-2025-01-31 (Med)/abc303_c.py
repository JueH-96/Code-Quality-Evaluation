def main():
    import sys
    input = sys.stdin.readline

    N, M, H, K = map(int, input().split())
    S = input().strip()
    
    # Use a set to store positions of the items for O(1) lookup.
    items = set()
    for _ in range(M):
        x, y = map(int, input().split())
        items.add((x, y))
    
    # Starting point (0, 0)
    x, y = 0, 0
    
    for move in S:
        # Each move consumes 1 health.
        H -= 1
        if move == 'R':
            x += 1
        elif move == 'L':
            x -= 1
        elif move == 'U':
            y += 1
        elif move == 'D':
            y -= 1
        
        # If health becomes negative, Takahashi collapses.
        if H < 0:
            print("No")
            return
        
        # If an item exists at the current position and health < K, then consume the item.
        if (x, y) in items:
            if H < K:
                H = K
                items.remove((x, y))
    
    print("Yes")

if __name__ == '__main__':
    main()