# YOUR CODE HERE
def solve():
    import sys
    from itertools import product

    N = int(sys.stdin.read())

    # Generate repunits up to 12 digits
    repunits = [int('1' * i) for i in range(1, 13)]

    sums = set()

    for r1, r2, r3 in product(repunits, repeat=3):
        sums.add(r1 + r2 + r3)

    sorted_sums = sorted(sums)

    print(sorted_sums[N-1])