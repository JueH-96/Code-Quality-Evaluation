import sys
import itertools

def main():
    # Read input 3x3 grid
    c = []
    for _ in range(3):
        c.extend(map(int, sys.stdin.readline().split()))
    # Define the 8 lines (rows, columns, diagonals) by their flat indices 0..8
    lines = [
        (0, 1, 2), (3, 4, 5), (6, 7, 8),  # rows
        (0, 3, 6), (1, 4, 7), (2, 5, 8),  # columns
        (0, 4, 8), (2, 4, 6)              # diagonals
    ]
    total = 0
    good = 0
    # Enumerate all permutations of the 9 cell‐indices
    for perm in itertools.permutations(range(9)):
        total += 1
        # time[p] = the step (0..8) at which cell p is seen
        time = [0]*9
        for t, p in enumerate(perm):
            time[p] = t
        # Check each line for a disappointment event
        bad = False
        for (a, b, d) in lines:
            ta, tb, td = time[a], time[b], time[d]
            # sort the three by seeing‐order
            # we only care when the third of the line is seen
            # whether the first two seen on that line are equal and differ from the third
            # so find which is first, second, third by time
            # we can do a small sort of tuples
            triple = sorted(((ta, a), (tb, b), (td, d)))
            # triple = [(t1, p1), (t2, p2), (t3, p3)]
            (t1, p1), (t2, p2), (t3, p3) = triple
            # if at their first two appearances same number, and third is different → disappointment
            if c[p1] == c[p2] and c[p1] != c[p3]:
                bad = True
                break
        if not bad:
            good += 1
    # Compute probability
    prob = good / total
    # Print with sufficient precision
    print("%.12f" % prob)

if __name__ == "__main__":
    main()