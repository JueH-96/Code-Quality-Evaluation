import sys

def main():
    input = sys.stdin.read
    data = input().split()

    N = int(data[0])
    M = int(data[1])
    positions = set()

    capture_offsets = [
        (2, 1), (1, 2), (-1, 2), (-2, 1),
        (-2, -1), (-1, -2), (1, -2), (2, -1)
    ]

    index = 2
    for _ in range(M):
        a, b = int(data[index]), int(data[index + 1])
        positions.add((a, b))
        index += 2

    captured_squares = set()

    for a, b in positions:
        for da, db in capture_offsets:
            x, y = a + da, b + db
            if 1 <= x <= N and 1 <= y <= N:
                captured_squares.add((x, y))

    total_squares = N * N
    occupied_squares = len(positions) + len(captured_squares)
    safe_squares = total_squares - occupied_squares

    print(safe_squares)

if __name__ == "__main__":
    main()