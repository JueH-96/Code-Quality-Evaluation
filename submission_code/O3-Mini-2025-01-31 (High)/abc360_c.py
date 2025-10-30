def main():
    import sys
    data = sys.stdin.read().strip().split()
    if not data:
        return
    it = iter(data)
    N = int(next(it))
    # Read initial positions for each item
    A = [int(next(it)) for _ in range(N)]
    # Read weights for each item
    W = [int(next(it)) for _ in range(N)]
    
    # Create buckets where index i represents box i (1-indexed)
    # Each bucket will store the weights of items originally in that box.
    boxes = [[] for _ in range(N+1)]
    for i in range(N):
        box_index = A[i]
        boxes[box_index].append(W[i])
    
    # For each box with one or more items,
    # we want to leave exactly one item (the heaviest to avoid high moving cost)
    # and move out all the other items.
    # The cost incurred from a box is sum(weights) - max(weight) if non-empty.
    # For boxes with exactly one item the cost is 0.
    total_cost = 0
    for box in boxes:
        if box:  # non-empty; note: box with one item gives (w - w) = 0
            # In an overloaded box, choose to keep the heaviest item.
            total_cost += sum(box) - max(box)
    
    sys.stdout.write(str(total_cost))
    
if __name__ == '__main__':
    main()