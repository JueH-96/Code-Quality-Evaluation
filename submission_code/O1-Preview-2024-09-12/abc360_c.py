# YOUR CODE HERE
import sys
import threading

def main():
    import sys
    sys.setrecursionlimit(1 << 25)
    N_and_rest = sys.stdin.read().split()
    N = int(N_and_rest[0])
    A = list(map(int, N_and_rest[1:N+1]))
    W = list(map(int, N_and_rest[N+1:2*N+1]))

    N_boxes = N + 1  # Since boxes are numbered from 1 to N
    box_items = [[] for _ in range(N_boxes)]  # box_items[box_number] = list of item indices in that box

    for idx in range(N):
        box = A[idx]
        box_items[box].append(idx)

    surplus_weights = []
    for box_idx in range(1, N_boxes):
        items_in_box = box_items[box_idx]
        num_items = len(items_in_box)
        if num_items > 1:
            # Find the item with maximal weight to keep in the box
            max_weight = -1
            item_to_keep = -1
            for item_idx in items_in_box:
                if W[item_idx] > max_weight:
                    max_weight = W[item_idx]
                    item_to_keep = item_idx
            # Collect the surplus items' weights
            for item_idx in items_in_box:
                if item_idx != item_to_keep:
                    surplus_weights.append(W[item_idx])
        # No need to process boxes with 0 or 1 item
        # Boxes with 0 items are counted in deficit_count later if needed

    # Count deficit boxes (boxes with no items)
    deficit_count = 0
    for box_idx in range(1, N_boxes):
        if len(box_items[box_idx]) == 0:
            deficit_count += 1

    # Total surplus items should equal deficit_count
    if len(surplus_weights) != deficit_count:
        # This should not happen as total surplus items should equal total deficits
        pass  # or handle error if necessary

    # Total cost is sum of surplus items' weights
    total_cost = sum(surplus_weights)
    print(total_cost)

threading.Thread(target=main).start()