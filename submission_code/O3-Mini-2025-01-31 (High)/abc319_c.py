def main():
    import sys, math, itertools
    data = sys.stdin.read().strip().split()
    if not data:
        return
    # Read the 3x3 grid in row‐major order (cells 0..8)
    vals = list(map(int, data))
    # The grid is indexed as follows:
    # row0: indices 0,1,2; row1: indices 3,4,5; row2: indices 6,7,8.
    #
    # Pre‐compute the 8 lines (3 rows, 3 columns, 2 diagonals)
    lines = [
        (0, 1, 2),  # row 1
        (3, 4, 5),  # row 2
        (6, 7, 8),  # row 3
        (0, 3, 6),  # col 1
        (1, 4, 7),  # col 2
        (2, 5, 8),  # col 3
        (0, 4, 8),  # main diagonal
        (2, 4, 6)   # anti-diagonal
    ]
    
    total = math.factorial(9)  # total number of orders = 9! = 362880
    valid_count = 0

    # Explanation:
    # Takahashi sees the nine cells in a random order. For any given line (three cells),
    # if the first two cells he sees are the ones with an identical number and the third cell
    # (revealed later) is different, he gets disappointed.
    # (Note: With the grid restrictions, each line either has all distinct numbers
    #  or exactly two equal and one different.)
    #
    # For each permutation (order in which cells are revealed) we compute an "order" array
    # such that order[i] is the reveal time (0-indexed) for cell i.
    # Then for each line, we determine which two cells come first and check if they have the same number
    # while the last cell is different.
    
    # Use local variables for speed.
    vals_local = vals
    lines_local = lines
    fact9 = total  # 362880

    for perm in itertools.permutations(range(9)):
        # Build the inverse mapping: for cell i, order[i] = reveal position in 'perm'
        order = [0] * 9
        for i, cell in enumerate(perm):
            order[cell] = i

        disappointed = False
        # Check each line
        for a, b, c in lines_local:
            ta = order[a]
            tb = order[b]
            tc = order[c]
            # We need to obtain the order of the three cells by their reveal time.
            # Since there are only three cells, we do it manually with comparisons.
            if ta < tb:
                if tb < tc:
                    f, s, t = a, b, c  # order: ta, tb, tc
                elif ta < tc:
                    f, s, t = a, c, b  # order: ta, tc, tb
                else:
                    f, s, t = c, a, b  # order: tc, ta, tb
            else:
                if ta < tc:
                    f, s, t = b, a, c  # order: tb, ta, tc
                elif tb < tc:
                    f, s, t = b, c, a  # order: tb, tc, ta
                else:
                    f, s, t = c, b, a  # order: tc, tb, ta

            # The disappointment condition: if the first two revealed cells have the same value
            # and the third (revealed last) has a different value.
            if vals_local[f] == vals_local[s] and vals_local[f] != vals_local[t]:
                disappointed = True
                break
        if not disappointed:
            valid_count += 1

    prob = valid_count / fact9
    # Print the probability with sufficient precision.
    sys.stdout.write("{:.12f}".format(prob))

if __name__ == '__main__':
    main()