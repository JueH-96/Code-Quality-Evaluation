import sys

def main():
    input = sys.stdin.read().split()
    ptr = 0
    N = int(input[ptr])
    ptr += 1
    M = int(input[ptr])
    ptr += 1

    occupied = set()
    positions = []
    for _ in range(M):
        a = int(input[ptr])
        ptr += 1
        b = int(input[ptr])
        ptr += 1
        occupied.add((a, b))
        positions.append((a, b))
    
    captured = set()
    for (a, b) in positions:
        moves = [
            (a + 2, b + 1),
            (a + 1, b + 2),
            (a - 1, b + 2),
            (a - 2, b + 1),
            (a - 2, b - 1),
            (a - 1, b - 2),
            (a + 1, b - 2),
            (a + 2, b - 1),
        ]
        for (x, y) in moves:
            if 1 <= x <= N and 1 <= y <= N:
                captured.add((x, y))
    
    X = 0
    for pos in occupied:
        if pos in captured:
            X += 1
    
    total_empty = N * N - M
    total_captured = len(captured)
    ans = total_empty - (total_captured - X)
    print(ans)

if __name__ == "__main__":
    main()