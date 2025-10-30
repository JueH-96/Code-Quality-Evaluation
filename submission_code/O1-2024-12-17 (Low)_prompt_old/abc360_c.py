def solve():
    import sys
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    A = list(map(int, data[1:1+N]))
    W = list(map(int, data[1+N:1+2*N]))

    # box_max[box] will hold the maximum weight among items in that box
    box_max = [0]*(N+1)  # Using 1-based indexing for convenience

    total_weight = 0
    for i in range(N):
        total_weight += W[i]
        box = A[i]
        if W[i] > box_max[box]:
            box_max[box] = W[i]

    # Sum of the maximum weights for boxes that have at least one item
    sum_of_max = sum(box_max)  # box_max[0] is unused

    # The minimum cost is total_weight - sum_of_kept_items
    # We keep one (the heaviest) item in each box that originally had items
    # and move the rest.
    answer = total_weight - sum_of_max
    print(answer)

def main():
    solve()

if __name__ == "__main__":
    main()