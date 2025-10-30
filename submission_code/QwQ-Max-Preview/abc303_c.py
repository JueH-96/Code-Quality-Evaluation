def main():
    import sys
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    M = int(input[ptr])
    ptr += 1
    H = int(input[ptr])
    ptr += 1
    K = int(input[ptr])
    ptr += 1
    S = input[ptr]
    ptr += 1

    items = set()
    for _ in range(M):
        x = int(input[ptr])
        ptr += 1
        y = int(input[ptr])
        ptr += 1
        items.add((x, y))
    
    current_x = 0
    current_y = 0
    health = H
    for c in S:
        if c == 'R':
            current_x += 1
        elif c == 'L':
            current_x -= 1
        elif c == 'U':
            current_y += 1
        elif c == 'D':
            current_y -= 1
        
        health -= 1
        if health < 0:
            print("No")
            return
        
        pos = (current_x, current_y)
        if pos in items and health < K:
            health = K
            items.remove(pos)
    
    print("Yes")

if __name__ == "__main__":
    main()