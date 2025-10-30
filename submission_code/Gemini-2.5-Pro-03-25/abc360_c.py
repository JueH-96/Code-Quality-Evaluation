# YOUR CODE HERE
import sys
from collections import defaultdict

def main():
    # Read the number of boxes and items, N
    N = int(sys.stdin.readline())
    
    # Read the initial box assignments A_1, ..., A_N for items 1, ..., N.
    # Input is a single line of space-separated integers.
    A = list(map(int, sys.stdin.readline().split()))
    
    # Read the weights W_1, ..., W_N for items 1, ..., N.
    # Input is a single line of space-separated integers.
    W = list(map(int, sys.stdin.readline().split()))

    # Use defaultdict to efficiently group item weights by their initial box number.
    # The key will be the box number (from 1 to N), and the value will be a list 
    # containing the weights of all items initially located in that box.
    # defaultdict(list) initializes the value to an empty list for a new key.
    boxes = defaultdict(list)
    
    # Accumulate the total weight of all items. This will be used later to calculate the cost.
    total_weight = 0
    
    # Iterate through all N items to populate the `boxes` defaultdict
    # and calculate the `total_weight`.
    # The loop iterates N times. Inside the loop, dictionary access and list append
    # operations take approximately constant time on average. Summation is O(1).
    # Total time complexity for this part is O(N).
    for i in range(N):
        # A[i] gives the box number for the (i+1)-th item (using 1-based indexing as in problem).
        box_idx = A[i]
        # W[i] gives the weight of the (i+1)-th item.
        weight = W[i]
        # Append the item's weight to the list associated with its initial box number.
        boxes[box_idx].append(weight)
        # Add the item's weight to the overall total weight sum.
        total_weight += weight

    # Calculate the sum of the maximum weights from each non-empty box.
    # The logic is: to minimize the cost of moving items, we should maximize the total weight
    # of items that *do not* move. An item can only stay in its initial box if it's the
    # *only* item that ends up in that box. If multiple items start in the same box,
    # at most one can stay. To maximize the saved weight (and thus minimize cost),
    # we should choose the item with the maximum weight in that box to potentially stay.
    # All other items in that box *must* move.
    sum_max_weights = 0
    
    # Iterate through the keys present in the `boxes` defaultdict.
    # These keys correspond exactly to the boxes that initially contain at least one item.
    # The number of keys is at most N.
    for box_idx in boxes: 
        # For each box that initially contains items (is non-empty), find the maximum weight among them.
        # Since we are iterating through the keys of `boxes`, `boxes[box_idx]` is guaranteed to be a non-empty list.
        # The `max()` function efficiently finds the maximum element in the list. If the list for box `k`
        # has size `c_k`, this takes O(c_k) time. Since the sum of `c_k` over all boxes is N, the total
        # time complexity for this loop is O(N).
        max_w_in_box = max(boxes[box_idx])
        
        # Add this maximum weight to `sum_max_weights`. This sum represents the maximum possible total weight
        # of items that could potentially stay in their original boxes.
        sum_max_weights += max_w_in_box
    
    # The minimum total cost required is the total weight of all items minus the maximum
    # total weight of items that can be saved by keeping them stationary.
    # The items that are not the maximum weight item in their respective initial boxes must be moved,
    # incurring a cost equal to their weight. The total minimum cost is the sum of weights of these
    # necessarily moved items. This value is precisely `total_weight - sum_max_weights`.
    # It has been reasoned that a configuration achieving this cost is always possible.
    min_cost = total_weight - sum_max_weights
    
    # Print the calculated minimum cost to standard output.
    print(min_cost)

# Standard practice in Python competitive programming: Check if the script is run directly
# (not imported as a module) and call the main function.
if __name__ == '__main__':
    main()