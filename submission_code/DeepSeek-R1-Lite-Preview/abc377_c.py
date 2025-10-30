import sys

def main():
    data = sys.stdin.read().split()
    ptr = 0
    N = int(data[ptr])
    ptr += 1
    M = int(data[ptr])
    ptr += 1
    pieces = set()
    for _ in range(M):
        a = int(data[ptr])
        ptr += 1
        b = int(data[ptr])
        ptr += 1
        pieces.add((a, b))
    moves = [
        (2, 1),
        (1, 2),
        (-1, 2),
        (-2, 1),
        (-2, -1),
        (-1, -2),
        (1, -2),
        (2, -1)
    ]
    threatened = set()
    for (i, j) in pieces:
        for (di, dj) in moves:
            ni, nj = i + di, j + dj
            if 1 <= ni <= N and 1 <= nj <= N and (ni, nj) not in pieces:
                threatened.add((ni, nj))
    total_empty = N * N - M
    safe_empty = total_empty - len(threatened)
    print(safe_empty)

if __name__ == '__main__':
    main()