def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    N = int(data[0])
    K = int(data[1])
    S = data[2]
    
    # Find all 1-blocks using 1-indexed positions.
    blocks = []
    i = 1
    while i <= N:
        if S[i - 1] == '1':
            # If this is the beginning of a 1-block
            if i == 1 or S[i - 2] == '0':
                start = i
                j = i
                while j <= N and S[j - 1] == '1':
                    j += 1
                end = j - 1
                blocks.append((start, end))
                i = j
                continue
        i += 1

    # According to the problem, S has at least K 1-blocks.
    # Let the (K-1)-th block (from the beginning) end at r_prev,
    # and let the K-th block start at lK and end at rK.
    lK, rK = blocks[K - 1]
    _, r_prev = blocks[K - 2]
    # Length of the K-th block.
    L = rK - lK + 1
    
    # We now build the new string T of length N.
    # T[1 ... r_prev] remains S[1 ... r_prev].
    # T[r_prev+1 ... r_prev+L] are turned into '1's.
    # T[r_prev+L+1 ... rK] are turned into '0's.
    # T[rK+1 ... N] remains S[rK+1 ... N].
    #
    # Converting to Python's 0-indexed slicing:
    # S[:r_prev] corresponds to S_1...S_{r_prev} (since r_prev is 1-indexed).
    # S[rK:] corresponds to S_{rK+1}...S_N.
    part1 = S[:r_prev]                  # unchanged first part (positions 1 to r_prev)
    part2 = '1' * L                     # the moved block inserted as all ones
    zeros_count = rK - (r_prev + L)       # positions to be filled with '0'
    part3 = '0' * zeros_count if zeros_count > 0 else ""
    part4 = S[rK:]                      # unchanged last part (positions rK+1 to N)
    
    T = part1 + part2 + part3 + part4
    sys.stdout.write(T)

if __name__ == '__main__':
    main()