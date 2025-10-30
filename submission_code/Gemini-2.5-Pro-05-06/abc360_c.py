import sys

def main():
    N = int(sys.stdin.readline())
    # A_values[i] is the box for item i (0-indexed).
    # Input A_i are 1-indexed box numbers.
    A_values = list(map(int, sys.stdin.readline().split()))
    # W_values[i] is the weight of item i (0-indexed).
    W_values = list(map(int, sys.stdin.readline().split()))

    # max_weight_for_box[j] will store the maximum weight of an item
    # initially located in box j+1. (Using 0-indexed array for boxes).
    # Box numbers in input are 1 to N.
    # Array indices for boxes will be 0 to N-1.
    max_weight_for_box = [0] * N 

    total_weight_of_all_items = 0
    for i in range(N):
        # Current item is item i (0-indexed).
        # Its initial box is A_values[i] (1-indexed).
        # Its weight is W_values[i].
        
        item_initial_box_one_indexed = A_values[i]
        item_weight = W_values[i]
        
        total_weight_of_all_items += item_weight
        
        # Convert box number to 0-indexed for array access.
        box_idx_zero_indexed = item_initial_box_one_indexed - 1
        
        # If this item is heavier than any other item seen so far for this box,
        # it becomes the candidate to be kept in this box (i.e., not moved).
        if item_weight > max_weight_for_box[box_idx_zero_indexed]:
            max_weight_for_box[box_idx_zero_indexed] = item_weight
            
    # total_weight_of_items_kept is the sum of weights of items we choose NOT to move.
    # For each box, we choose not to move the heaviest item that was initially in it (if any).
    # All other items initially in that same box must be moved.
    # Items that were in initially empty boxes must be filled by moved items.
    total_weight_of_items_kept = 0
    for weight in max_weight_for_box:
        total_weight_of_items_kept += weight
        
    # The cost is the sum of weights of items that ARE moved.
    # This is equivalent to (total_weight_of_all_items - total_weight_of_items_kept).
    min_total_cost = total_weight_of_all_items - total_weight_of_items_kept
    
    sys.stdout.write(str(min_total_cost) + "
")

if __name__ == '__main__':
    main()