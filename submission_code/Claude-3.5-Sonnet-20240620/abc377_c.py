# YOUR CODE HERE
def can_capture(x1, y1, x2, y2):
    dx = abs(x1 - x2)
    dy = abs(y1 - y2)
    return (dx == 2 and dy == 1) or (dx == 1 and dy == 2)

def main():
    N, M = map(int, input().split())
    pieces = set()
    captured = set()

    for _ in range(M):
        a, b = map(int, input().split())
        pieces.add((a, b))
        
        for dx in [-2, -1, 1, 2]:
            for dy in [-2, -1, 1, 2]:
                if can_capture(0, 0, dx, dy):
                    x, y = a + dx, b + dy
                    if 1 <= x <= N and 1 <= y <= N:
                        captured.add((x, y))

    total_squares = N * N
    unavailable = len(pieces) + len(captured)
    available = total_squares - unavailable

    print(available)

if __name__ == "__main__":
    main()