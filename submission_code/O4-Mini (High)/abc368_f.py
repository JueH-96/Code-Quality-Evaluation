import sys

def main():
    data = sys.stdin.buffer.read().split()
    it = iter(data)
    try:
        n = int(next(it))
    except StopIteration:
        return
    A = [int(x) for x in it]
    if n == 0:
        print("Bruno")
        return
    maxA = max(A)
    # bits[x] will be a bitmask of reachable Grundy values from proper divisors of x
    bits = [0] * (maxA + 1)
    # g[x] will be the Grundy number for x
    g = [0] * (maxA + 1)
    # Compute g[1..maxA]
    for v in range(1, maxA + 1):
        mask = bits[v]
        j = 0
        # find mex: smallest j with bit j not set in mask
        while mask & (1 << j):
            j += 1
        g[v] = j
        # propagate this Grundy value to all multiples of v (proper divisors)
        shift = 1 << j
        step = v
        mv = v + v
        # for each multiple m of v greater than v
        while mv <= maxA:
            bits[mv] |= shift
            mv += step

    # compute XOR of all piles
    xorsum = 0
    for ai in A:
        xorsum ^= g[ai]

    # Anna wins if XOR != 0
    print("Anna" if xorsum != 0 else "Bruno")

if __name__ == "__main__":
    main()