import sys

def main():
    data = sys.stdin.read().splitlines()
    if not data:
        print("No")
        return
    
    parts = data[0].split()
    N = int(parts[0])
    M = int(parts[1])
    H = int(parts[2])
    K = int(parts[3])
    
    S = data[1].strip()
    
    items = set()
    for i in range(2, 2 + M):
        coords = data[i].split()
        if len(coords) < 2:
            continue
        x = int(coords[0])
        y = int(coords[1])
        items.add((x, y))
    
    x, y = 0, 0
    health = H
    
    for c in S:
        if c == 'R':
            x += 1
        elif c == 'L':
            x -= 1
        elif c == 'U':
            y += 1
        elif c == 'D':
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