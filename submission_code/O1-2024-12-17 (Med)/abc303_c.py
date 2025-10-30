def main():
    import sys
    data = sys.stdin.read().strip().split()
    N, M, H, K = map(int, data[:4])
    S = data[4]
    coords = data[5:]
    
    items = set()
    idx = 0
    for _ in range(M):
        x = int(coords[idx]); y = int(coords[idx+1])
        items.add((x,y))
        idx += 2

    x, y = 0, 0
    health = H
    
    for move in S:
        if move == 'R':
            x += 1
        elif move == 'L':
            x -= 1
        elif move == 'U':
            y += 1
        elif move == 'D':
            y -= 1
        
        health -= 1
        if health < 0:
            print("No")
            return
        
        if (x, y) in items and health < K:
            health = K
            items.remove((x, y))
    
    print("Yes")

if __name__ == "__main__":
    main()