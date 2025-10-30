def main():
    import sys
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    A = list(map(int, data[1:1+N]))
    W = list(map(int, data[1+N:1+2*N]))

    # We will store, for each box, the maximum weight of any item in it.
    # Initialize with -1 to indicate an empty box (no items yet).
    box_max = [-1] * N

    # Fill in the maximum weight for each box
    for i in range(N):
        box_index = A[i] - 1  # zero-based
        if W[i] > box_max[box_index]:
            box_max[box_index] = W[i]

    # Sum of all item weights
    total_weight = sum(W)

    # Sum of maximum weights in each box that is not empty (-1 means empty)
    sum_max = sum(m for m in box_max if m != -1)

    # The minimum cost is total_weight - sum_of_max_in_each_non_empty_box
    print(total_weight - sum_max)

# Do not forget to call main
if __name__ == "__main__":
    main()