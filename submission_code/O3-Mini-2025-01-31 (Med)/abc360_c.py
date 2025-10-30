def main():
    import sys
    data = sys.stdin.read().strip().split()
    if not data:
        return
    it = iter(data)
    n = int(next(it))
    
    # Read the initial box for each item.
    A = [int(next(it)) for _ in range(n)]
    # Read the weight for each item.
    W = [int(next(it)) for _ in range(n)]
    
    # Total weight of all items.
    total_weight = sum(W)
    
    # For each box (1-indexed, so we use 0-indexed storage), we record the maximum weight
    # of an item that is originally in that box.
    max_in_box = [0] * n  # n boxes
    for a, w in zip(A, W):
        # Convert box number to 0-indexed.
        idx = a - 1
        if w > max_in_box[idx]:
            max_in_box[idx] = w

    # In each box that originally had items, we can keep the heaviest one to avoid its moving cost.
    # Thus, the cost incurred in a box with items is the sum of the weights of the box minus the max.
    # Over all boxes, the total moving cost is:
    #   total_weight - (sum of all kept item weights).
    # Note: For boxes with no items, max_in_box remains 0.
    ans = total_weight - sum(max_in_box)
    sys.stdout.write(str(ans))
    
if __name__ == '__main__':
    main()