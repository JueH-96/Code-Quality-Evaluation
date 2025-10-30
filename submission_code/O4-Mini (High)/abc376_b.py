import sys

def main():
    input = sys.stdin.readline
    N, Q = map(int, input().split())
    # Use 0-based indexing for parts 1..N -> 0..N-1
    left = 0
    right = 1
    total_moves = 0

    for _ in range(Q):
        line = input().split()
        H = line[0]
        T = int(line[1]) - 1  # target in 0..N-1

        if H == 'L':
            a, b, forbidden = left, T, right
        else:
            a, b, forbidden = right, T, left

        # Compute distances avoiding the forbidden node
        # da = clockwise distance from a to b
        da = (b - a) % N
        # db = counter-clockwise distance from a to b
        db = (a - b) % N
        # position of forbidden relative to a in cw/ccw
        df_cw = (forbidden - a) % N
        df_ccw = (a - forbidden) % N

        best = 10**18
        # clockwise path is valid if forbidden is not strictly between a and b on that arc
        if not (0 < df_cw <= da):
            best = da
        # counter-clockwise path is valid if forbidden is not strictly between a and b on that arc
        if not (0 < df_ccw <= db):
            best = min(best, db)

        total_moves += best

        # update the moved hand
        if H == 'L':
            left = T
        else:
            right = T

    print(total_moves)

if __name__ == '__main__':
    main()