def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    N = int(data[0])
    K = int(data[1])
    S = data[2].strip()
    
    # Identify all 1-blocks in S.
    # A 1-block is a maximal contiguous substring of '1's,
    # where its left neighbor (if exists) is '0' and its right neighbor (if exists) is '0'.
    blocks = []
    in_block = False
    for i, ch in enumerate(S):
        if ch == '1':
            if not in_block:
                start = i  # start index of the current block (0-indexed)
                in_block = True
        else:  # ch == '0'
            if in_block:
                blocks.append((start, i - 1))
                in_block = False
    # If the string ends with a block of 1's, add it.
    if in_block:
        blocks.append((start, N - 1))
    
    # We are guaranteed that S contains at least K 1-blocks, and K >= 2.
    # Let the (K-1)-th 1-block (1-indexed) be A and the K-th 1-block be B.
    # In our zero-index representation:
    # A is blocks[K-2] and B is blocks[K-1].
    A = blocks[K - 2]
    B = blocks[K - 1]
    
    # Let's convert the borders to 1-indexed positions for clarity:
    # In the problem specification:
    #   Block A is from positions l_A to r_A, where r_A = A[1] + 1 (since our blocks are 0-indexed).
    #   Block B is from positions l_B to r_B, where l_B = B[0] + 1 and r_B = B[1] + 1.
    # According to the problem, we must:
    #   - Copy S[1 ... r_A] unchanged.
    #   - Replace the segment from r_A + 1 to r_A + (length of block B) with ones.
    #   - Replace the segment from r_A + (length of block B) + 1 to r_B with zeros.
    #   - Copy S[r_B + 1 ... N] unchanged.
    # For Python slicing (0-indexed), we note:
    #   * The prefix S[:A_end] corresponds to positions 1 through r_A in 1-indexed,
    #     with A_end = A[1] + 1.
    #   * Block B’s length is B_len = (B[1] - B[0] + 1).
    #   * The end of block B in S (1-indexed) is B_end = B[1] + 1.
    A_end = A[1] + 1
    B_len = B[1] - B[0] + 1
    B_end = B[1] + 1

    # Construct the new string T:
    #   T[0 : A_end]           = S[:A_end]
    #   Next B_len positions    = "1" * B_len
    #   Next (B_end - A_end - B_len) positions = "0" * (B_end - A_end - B_len)
    #   The tail                 = S[B_end:]
    part1 = S[:A_end]
    part2 = "1" * B_len
    part3 = "0" * (B_end - A_end - B_len)
    part4 = S[B_end:]
    
    result = part1 + part2 + part3 + part4
    sys.stdout.write(result)

if __name__ == '__main__':
    main()