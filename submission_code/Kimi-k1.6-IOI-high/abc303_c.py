import sys

def main():
    n, m, h, k = map(int, sys.stdin.readline().split())
    s = sys.stdin.readline().strip()
    items = set()
    for _ in range(m):
        x, y = map(int, sys.stdin.readline().split())
        items.add((x, y))
    
    current_x, current_y = 0, 0
    current_health = h
    
    for c in s:
        # Move according to the direction
        if c == 'R':
            current_x += 1
        elif c == 'L':
            current_x -= 1
        elif c == 'U':
            current_y += 1
        elif c == 'D':
            current_y -= 1
        
        # Consume health
        current_health -= 1
        if current_health < 0:
            print("No")
            return
        
        # Check for item consumption
        if (current_x, current_y) in items and current_health < k:
            items.remove((current_x, current_y))
            current_health = k
    
    print("Yes")

if __name__ == "__main__":
    main()