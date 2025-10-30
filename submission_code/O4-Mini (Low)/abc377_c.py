import sys
import threading

def main():
    import sys
    data = sys.stdin.read().split()
    N = int(data[0])
    M = int(data[1])
    coords = data[2:]
    # Read occupied positions
    occ = set()
    for i in range(0, 2*M, 2):
        a = int(coords[i])
        b = int(coords[i+1])
        # encode pair into single integer to speed up lookups
        occ.add((a << 32) | b)

    # Possible knight moves
    moves = [(2,1),(1,2),(-1,2),(-2,1),(-2,-1),(-1,-2),(1,-2),(2,-1)]

    attacked = set()
    # For each occupied piece, mark its attack targets
    for code in occ:
        # decode
        a = code >> 32
        b = code & 0xFFFFFFFF
        for dx, dy in moves:
            x = a + dx
            y = b + dy
            # must lie inside the board
            if 1 <= x <= N and 1 <= y <= N:
                c2 = (x << 32) | y
                # only count if not already occupied
                if c2 not in occ:
                    attacked.add(c2)

    # total squares = N^2
    total = N * N
    # empty squares = total - M
    # attacked empty = len(attacked)
    result = total - M - len(attacked)
    # print result
    sys.stdout.write(str(result))

if __name__ == "__main__":
    main()