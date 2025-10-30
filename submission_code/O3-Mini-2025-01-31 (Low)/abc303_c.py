def main():
    import sys

    input = sys.stdin.readline
    N, M, H, K = map(int, input().split())
    S = input().strip()

    # Insert items into a set for O(1) lookup
    items = set()
    for _ in range(M):
        x, y = map(int, input().split())
        items.add((x, y))
    
    # starting position
    x, y = 0, 0
    
    for move in S:
        # decrease health by 1
        H -= 1
        if move == 'R':
            x += 1
        elif move == 'L':
            x -= 1
        elif move == 'U':
            y += 1
        elif move == 'D':
            y -= 1
        
        # if health less than 0, he collapses
        if H < 0:
            print("No")
            return
        
        # if there's an item at the new position and H is less than K, use the item
        if (x, y) in items and H < K:
            H = K
            items.remove((x, y))
    
    print("Yes")

if __name__ == '__main__':
    main()