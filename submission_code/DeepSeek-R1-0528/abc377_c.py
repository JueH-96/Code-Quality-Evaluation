import sys

def main():
    data = sys.stdin.read().split()
    if not data:
        print(0)
        return
    it = iter(data)
    N = int(next(it))
    M = int(next(it))
    pieces = []
    occupied = set()
    for _ in range(M):
        a = int(next(it))
        b = int(next(it))
        pieces.append((a, b))
        occupied.add((a, b))
    
    moves = [
        (2, 1), (1, 2), (-1, 2), (-2, 1),
        (-2, -1), (-1, -2), (1, -2), (2, -1)
    ]
    attacked = set()
    for a, b in pieces:
        for dx, dy in moves:
            x = a + dx
            y = b + dy
            if 1 <= x <= N and 1 <= y <= N:
                attacked.add((x, y))
    
    total_forbidden = occupied.union(attacked)
    ans = N * N - len(total_forbidden)
    print(ans)

if __name__ == '__main__':
    main()