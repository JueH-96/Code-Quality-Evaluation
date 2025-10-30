import sys
import threading

def main():
    data = sys.stdin
    line = data.readline().split()
    if not line:
        return
    N = int(line[0])
    M = int(line[1])

    # Read occupied squares, encode each as a single integer (r << 32) | c
    occ = set()
    for _ in range(M):
        a, b = map(int, data.readline().split())
        occ.add((a << 32) | b)

    # Knight moves: 8 possible offsets
    moves = [(-2, -1), (-2, 1), (-1, -2), (-1, 2),
             (1, -2), (1, 2), (2, -1), (2, 1)]

    attacked = set()
    # For each occupied square, mark its attacked neighbors (if in bounds & not occupied)
    for code in occ:
        # decode row, col
        r = code >> 32
        c = code & 0xFFFFFFFF
        for dr, dc in moves:
            rr = r + dr
            cc = c + dc
            if 1 <= rr <= N and 1 <= cc <= N:
                e = (rr << 32) | cc
                if e not in occ:
                    attacked.add(e)

    # Total squares = N*N, occupied = M, attacked_empty = len(attacked)
    total = N * N
    result = total - M - len(attacked)
    print(result)

if __name__ == "__main__":
    main()