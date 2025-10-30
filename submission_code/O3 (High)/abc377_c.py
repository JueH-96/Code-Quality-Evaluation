import sys

# Offsets a piece can capture (knight moves)
MOVES = (
    ( 2,  1), ( 1,  2), (-1,  2), (-2,  1),
    (-2, -1), (-1, -2), ( 1, -2), ( 2, -1),
)

def main():
    data = sys.stdin.buffer.read().split()
    if not data:
        return
    it = iter(data)
    N = int(next(it))
    M = int(next(it))

    pieces = set()
    for _ in range(M):
        a = int(next(it))
        b = int(next(it))
        pieces.add((a, b))

    threatened = set()

    for a, b in pieces:
        for da, db in MOVES:
            na, nb = a + da, b + db
            if 1 <= na <= N and 1 <= nb <= N:
                pos = (na, nb)
                if pos not in pieces:
                    threatened.add(pos)

    total_squares = N * N
    safe = total_squares - M - len(threatened)
    print(safe)

if __name__ == "__main__":
    main()