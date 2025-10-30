import sys

def solve():
    N = int(sys.stdin.readline())
    A = list(map(int, sys.stdin.readline().split()))
    W = list(map(int, sys.stdin.readline().split()))

    # Group items by their initial box
    # box_items = {box_id: [(weight, item_id), ...]}
    # Using 1-based indexing for box_id to match problem statement
    box_items = {}
    for i in range(N):
        box_id = A[i]
        weight = W[i]
        item_id = i + 1 # Use 1-based item index
        if box_id not in box_items:
            box_items[box_id] = []
        box_items[box_id].append((weight, item_id))

    total_cost = 0

    # Iterate through each box that initially contains items
    for box_id in box_items:
        items_in_box = box_items[box_id]

        if not items_in_box:
            # Should not happen based on how box_items is populated, but as a check
            continue

        # Calculate the sum of weights of all items in this box
        sum_weights_in_box = sum(weight for weight, item_id in items_in_box)

        # Find the maximum weight in this box
        max_weight = -1
        for weight, item_id in items_in_box:
            if weight > max_weight:
                max_weight = weight

        # Count how many items have this maximum weight
        max_count = 0
        for weight, item_id in items_in_box:
            if weight == max_weight:
                max_count += 1

        if max_count == 1:
            # If there is a unique item with maximum weight, that item can potentially stay in this box
            # without incurring cost for itself. The cost for this box is the sum of weights
            # of all other items in this box (which must be moved out).
            total_cost += sum_weights_in_box - max_weight
        else:
            # If there are multiple items with maximum weight (or no items, although handled by the loop)
            # there is no unique heaviest item that can be preferentially kept cost-free.
            # All items initially in this box contribute to the cost of needing rearrangement involving this box.
            # The minimum cost associated with clearing this box and potentially bringing in another item
            # seems to require moving all items out.
            total_cost += sum_weights_in_box

    print(total_cost)

solve()