import sys
import itertools

def main() -> None:
    # read grid and flatten to 1-D list of 9 values
    c = []
    for _ in range(3):
        c.extend(map(int, sys.stdin.readline().split()))
    # all 8 relevant lines expressed with flattened indices
    lines = [
        (0, 1, 2), (3, 4, 5), (6, 7, 8),        # rows
        (0, 3, 6), (1, 4, 7), (2, 5, 8),        # columns
        (0, 4, 8), (2, 4, 6)                    # diagonals
    ]

    total_perm = 362880   # 9!
    good_perm = 0

    for perm in itertools.permutations(range(9)):
        # build rank array: rank[pos] = when that position is seen
        rank = [0]*9
        for t, pos in enumerate(perm):
            rank[pos] = t

        ok = True
        for a, b, d in lines:
            # obtain first two squares in temporal order
            triple = [(rank[a], a), (rank[b], b), (rank[d], d)]
            triple.sort(key=lambda x: x[0])
            p1, p2 = triple[0][1], triple[1][1]
            p3 = triple[2][1]

            if c[p1] == c[p2] and c[p1] != c[p3]:
                ok = False
                break

        if ok:
            good_perm += 1

    prob = good_perm / total_perm
    print("{:.15f}".format(prob))

if __name__ == "__main__":
    main()