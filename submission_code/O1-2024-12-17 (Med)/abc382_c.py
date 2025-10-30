def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    N, M = map(int, input_data[:2])
    A = list(map(int, input_data[2:2+N]))
    B = list(map(int, input_data[2+N:2+N+M]))

    # A segment tree to store the minimum value in each segment
    # We'll build it so we can quickly find the leftmost index i where A[i] <= x.

    # 1) Prepare size for segment tree (power of two >= N).
    size = 1
    while size < N:
        size <<= 1

    # 2) Initialize tree with a large "infinity" sentinel.
    INF = 10**9 + 1
    seg = [INF] * (2 * size)  # store the min of a range

    # 3) Build: fill leaves
    for i in range(N):
        seg[size + i] = A[i]
    # build internal nodes
    for i in range(size - 1, 0, -1):
        seg[i] = min(seg[2 * i], seg[2 * i + 1])

    # 4) Define a function to find the leftmost index i such that A[i] <= x.
    #    Returns -1 if none such index.
    def find_leftmost(x):
        # If the min value over the entire range is > x, no valid index
        if seg[1] > x:
            return -1

        # Otherwise, descend the tree starting from the root (index 1)
        idx = 1
        left, right = 0, size  # range covered by the root
        while right - left > 1:  # while we're not at a leaf
            mid = (left + right) // 2
            # Check the min in the left child
            if seg[idx * 2] <= x:
                # go left
                idx = idx * 2
                right = mid
            else:
                # go right
                idx = idx * 2 + 1
                left = mid

        # Here, left is the index of the leaf
        if left >= N:
            return -1
        return left

    # 5) For each sushi B_j, find the earliest person who can eat it.
    out = []
    for deliciousness in B:
        eater = find_leftmost(deliciousness)
        if eater == -1:
            out.append("-1")
        else:
            out.append(str(eater + 1))

    # 6) Print the results
    print("
".join(out))

# Don't forget to call main!
if __name__ == "__main__":
    main()