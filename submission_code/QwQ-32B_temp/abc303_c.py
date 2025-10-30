import sys

def main():
    N, M, H, K = map(int, sys.stdin.readline().split())
    S = sys.stdin.readline().strip()
    items = set()
    for _ in range(M):
        x, y = map(int, sys.stdin.readline().split())
        items.add((x, y))
    
    x, y = 0, 0
    health = H
    for c in S:
        # Update position based on direction
        if c == 'R':
            x += 1
        elif c == 'L':
            x -= 1
        elif c == 'U':
            y += 1
        else:  # D
            y -= 1
        
        # Consume health
        health -= 1
        if health < 0:
            print("No")
            return
        
        # Check for item
        current_pos = (x, y)
        if current_pos in items and health < K:
            health = K
            items.remove(current_pos)
    
    print("Yes")

if __name__ == "__main__":
    main()