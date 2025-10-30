def main():
    import sys

    # Read L and R from input
    L_str, R_str = sys.stdin.read().split()
    L = int(L_str)
    R = int(R_str)

    blocks = []
    l = L

    while l < R:
        if l == 0:
            # For l=0, find the largest i where 2^i <= R
            i_max = R.bit_length() - 1
        else:
            # Find the exponent of the least significant 1-bit in l
            i_max = (l & -l).bit_length() - 1

        # Find the largest i such that l + 2^i <= R
        i = i_max
        while (1 << i) > R - l:
            i -= 1

        # Add the block
        r = l + (1 << i)
        blocks.append((l, r))
        l = r

    # Output the result
    M = len(blocks)
    print(M)
    for block in blocks:
        print(block[0], block[1])

if __name__ == "__main__":
    main()