import sys

def main():
    data = sys.stdin.read().splitlines()
    if not data:
        return
    
    first_line = data[0].split()
    n = int(first_line[0])
    m = int(first_line[1])
    H = int(first_line[2])
    K = int(first_line[3])
    S = data[1].strip()
    
    items = set()
    for i in range(2, 2 + m):
        x, y = map(int, data[i].split())
        items.add((x, y))
    
    x, y = 0, 0
    health = H
    
    for i in range(n):
        c = S[i]
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
        
        if health < K and (x, y) in items:
            health = K
            items.remove((x, y))
    
    print("Yes")

if __name__ == "__main__":
    main()