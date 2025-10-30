def main():
    import sys
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    K = int(data[1])
    S = data[2]

    # Find the 1-blocks in S (0-based indexing).
    blocks = []
    in_block = False
    start = -1
    for i, ch in enumerate(S):
        if ch == '1':
            if not in_block:
                in_block = True
                start = i
        else:
            if in_block:
                blocks.append((start, i - 1))
                in_block = False
    if in_block:
        blocks.append((start, N - 1))

    # Identify the (K-1)-th block and K-th block (using 0-based for Python lists).
    l_k_1, r_k_1 = blocks[K - 2]
    l_k, r_k = blocks[K - 1]

    # Length of the K-th block.
    length_k = r_k - l_k + 1

    # Convert S to a list to modify.
    t = list(S)

    # According to the problem's formula (in 1-based),
    # T_i = S_i for i ≤ r_{K-1}
    # T_i = 1 for r_{K-1}+1 ≤ i ≤ r_{K-1} + length(kth block)
    # T_i = 0 for the next segment up to r_k
    # T_i = S_i beyond r_k
    # We'll do this in 0-based carefully.

    # Overwrite the region [r_k_1+1, r_k_1+1+length_k - 1] with '1'
    for i in range(r_k_1 + 1, r_k_1 + 1 + length_k):
        t[i] = '1'
    # Overwrite the region [r_k_1+1+length_k, r_k] with '0'
    for i in range(r_k_1 + 1 + length_k, r_k + 1):
        t[i] = '0'

    # Output the result
    print("".join(t))

# Do not forget to call main()
if __name__ == "__main__":
    main()