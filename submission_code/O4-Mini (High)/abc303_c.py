import sys

def main():
    input = sys.stdin.readline
    N, M, H, K = map(int, input().split())
    S = input().rstrip()
    
    items = set()
    for _ in range(M):
        x, y = map(int, input().split())
        items.add((x, y))
    
    x = 0
    y = 0
    health = H
    
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
        
        # Consume 1 health for the move
        health -= 1
        if health < 0:
            print("No")
            return
        
        # If health is below K and there's an item here, recover to K
        if health < K and (x, y) in items:
            items.remove((x, y))
            health = K
    
    print("Yes")

if __name__ == "__main__":
    main()