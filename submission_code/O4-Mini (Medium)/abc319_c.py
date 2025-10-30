import sys
import threading

def main():
    import sys
    from itertools import permutations

    data = sys.stdin.read().split()
    # Flatten the 3x3 grid into a list c of length 9 (row-major order)
    c = list(map(int, data))
    # Define the 8 lines (rows, columns, diagonals) by their 0-based indices
    lines = [
        (0, 1, 2),  # row 1
        (3, 4, 5),  # row 2
        (6, 7, 8),  # row 3
        (0, 3, 6),  # col 1
        (1, 4, 7),  # col 2
        (2, 5, 8),  # col 3
        (0, 4, 8),  # main diag
        (2, 4, 6),  # anti diag
    ]

    total = 0
    good = 0
    # Iterate over all permutations of the 9 positions (0..8)
    for perm in permutations(range(9)):
        # perm is a tuple of length 9: the viewing order of positions
        # Build an inverse map: at what time each position is seen
        t = [0]*9
        for time, pos in enumerate(perm):
            t[pos] = time

        # Check all lines for disappointment
        disappointed = False
        for (i, j, k) in lines:
            ti, tj, tk = t[i], t[j], t[k]
            # find the order in which i,j,k are seen
            # we want p1, p2 being the two smallest times, p3 the largest
            # pack and sort by time
            trio = sorted(((ti, i), (tj, j), (tk, k)))
            _, p1 = trio[0]
            _, p2 = trio[1]
            _, p3 = trio[2]
            # if the first two seen have equal value but the third differs
            if c[p1] == c[p2] and c[p2] != c[p3]:
                disappointed = True
                break

        total += 1
        if not disappointed:
            good += 1

    # Probability = good / total
    prob = good / total
    # Print with sufficient precision
    print("{:.12f}".format(prob))

if __name__ == "__main__":
    main()