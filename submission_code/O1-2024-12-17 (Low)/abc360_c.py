def main():
    import sys
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    A = list(map(int, data[1:1+N]))
    W = list(map(int, data[1+N:1+2*N]))
    
    # Sums and max-tracking per box (1-based indexing)
    box_sum = [0]*(N+1)
    box_max = [0]*(N+1)
    
    # Gather weights per box
    for i in range(N):
        box = A[i]
        w = W[i]
        box_sum[box] += w
        if w > box_max[box]:
            box_max[box] = w
    
    # Calculate cost: for each box that has items,
    # cost = sum_of_weights_in_box - max_weight_in_box
    # (boxes with no items contribute 0)
    total_cost = 0
    for b in range(1, N+1):
        if box_sum[b] > 0:
            total_cost += box_sum[b] - box_max[b]
    
    print(total_cost)

# Don't forget to call main!
if __name__ == "__main__":
    main()