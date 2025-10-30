import sys

def main():
    data = sys.stdin.read().split()
    ptr = 0
    N = int(data[ptr])
    ptr += 1
    M = int(data[ptr])
    ptr += 1
    H = int(data[ptr])
    ptr += 1
    K = int(data[ptr])
    ptr += 1
    S = data[ptr]
    ptr += 1
    items = set()
    for _ in range(M):
        x = int(data[ptr])
        ptr += 1
        y = int(data[ptr])
        ptr += 1
        items.add((x, y))
    # Initialize position
    x, y = 0, 0
    # Initialize health
    health = H
    # Define move mappings
    move_map = {
        'R': (1, 0),
        'L': (-1, 0),
        'U': (0, 1),
        'D': (0, -1)
    }
    for move in S:
        dx, dy = move_map[move]
        x += dx
        y += dy
        health -= 1
        if health < 0:
            print("No")
            return
        if (x, y) in items and health < K:
            health = K
    print("Yes")

if __name__ == '__main__':
    main()