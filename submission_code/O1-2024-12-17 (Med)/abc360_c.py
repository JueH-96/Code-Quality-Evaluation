def main():
    import sys
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    A = list(map(int, data[1:1+N]))
    W = list(map(int, data[1+N:1+2*N]))

    # Total weight of all items
    total_weight = sum(W)

    # For each box, track the maximum weight among items placed there
    max_in_box = [0]*(N+1)
    for i in range(N):
        box = A[i]
        weight = W[i]
        if weight > max_in_box[box]:
            max_in_box[box] = weight

    # Sum of max weights chosen to stay in their original box
    sum_keep = sum(max_in_box)

    # Minimum cost = total weight - total weight of items we can keep
    print(total_weight - sum_keep)

# Do not remove the following line
main()