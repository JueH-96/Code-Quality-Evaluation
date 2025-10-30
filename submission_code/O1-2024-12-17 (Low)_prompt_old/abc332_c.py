def solve():
    import sys
    data = sys.stdin.read().strip().split()
    N, M = map(int, data[:2])
    S = data[2]

    # We will split the string S into blocks of consecutive days where S[i] != '0'.
    # Each block ends whenever we encounter '0' or the string ends.
    # For each block, let:
    #   a = number of days with '1' (meal days),
    #   b = number of days with '2' (event days).
    #
    # Within one block (no wash day in between), we need:
    #   1) At least b logo T-shirts, because on '2' days only logo T-shirts can be worn.
    #   2) Enough T-shirts in total to cover all a + b days. We have M plain T-shirts plus
    #      the X logo T-shirts we might buy. So M + X >= a + b.
    #
    # Combining:
    #   X >= b  (must have enough logos for event days)
    #   X >= (a + b) - M  (must have enough total if that sum exceeds M)
    #
    # For each block, the required X must satisfy the above, hence for each block:
    #   needed_for_block = max(b, max((a + b) - M, 0))
    #
    # The total X we need is the maximum of needed_for_block over all blocks because
    # T-shirts get washed (reset) on '0' days, so the blocks are independent.
    #
    # Finally, if the result is negative, we print 0. Otherwise, we print that result.

    blocks = []
    a = b = 0
    in_block = False

    for c in S:
        if c == '0':
            # End the current block, if any
            if in_block:
                blocks.append((a, b))
                a = b = 0
                in_block = False
        else:
            in_block = True
            if c == '1':
                a += 1
            else:  # c == '2'
                b += 1

    # If the last block extends to the end, add it
    if in_block:
        blocks.append((a, b))

    answer = 0
    for a, b in blocks:
        # X must satisfy both constraints: X >= b and X >= (a + b) - M
        # Since X can't be negative, compare with 0 as well.
        needed = max(b, (a + b) - M, 0)
        answer = max(answer, needed)

    print(answer if answer > 0 else 0)

def main():
    solve()

if __name__ == "__main__":
    main()